"""
RTMP streaming service – framework-agnostic.

Pushes raw video frames to an RTMP endpoint via FFmpeg.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from typing import Callable, Optional


class StreamingService:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        rtmp_url: str,
        queue_size: int = 10,
    ) -> None:
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._rtmp_url = rtmp_url
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._running_checker: Optional[Callable[[], bool]] = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def start(self, fps: float, is_running: Callable[[], bool]) -> None:
        if not self._enabled or not self._rtmp_url:
            return

        self._running_checker = is_running
        command = [
            "ffmpeg",
            "-re",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._frame_width}x{self._frame_height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-f", "flv",
            self._rtmp_url,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"FFmpeg start failed: {exc}")
            self._process = None
            self._enabled = False
            return

        time.sleep(0.2)
        if self._process.poll() is not None:
            print("FFmpeg exited early. Live stream disabled.")
            self._process = None
            self._enabled = False
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, frame) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def toggle(
        self, enabled: bool, fps: float, is_running: Callable[[], bool]
    ) -> None:
        self._enabled = enabled
        if not enabled:
            self.stop()
            return
        if is_running() and self._process is None:
            self.start(fps, is_running)

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
        except Exception:
            pass
        self._process = None

    def _worker(self) -> None:
        while self._running_checker and self._running_checker():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if self._process is None:
                    continue
                if self._process.poll() is not None:
                    continue
                self._process.stdin.write(frame.tobytes())
            except Exception as exc:
                print(f"Error writing to ffmpeg stdin: {exc}")
