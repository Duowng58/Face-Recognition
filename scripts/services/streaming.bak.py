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
    _RESTART_DELAYS = (1, 2, 5, 10, 15, 30)  # exponential back-off seconds

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        rtmp_url: str,
        queue_size: int = 30,
    ) -> None:
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._rtmp_url = rtmp_url
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._running_checker: Optional[Callable[[], bool]] = None
        self._fps: float = 25.0
        self._restart_lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def start(self, fps: float, is_running: Callable[[], bool]) -> None:
        if not self._enabled or not self._rtmp_url:
            return

        self._fps = fps
        self._running_checker = is_running

        if not self._spawn_ffmpeg():
            return

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _spawn_ffmpeg(self) -> bool:
        """Launch (or re-launch) the FFmpeg subprocess. Returns True on success."""
        # Clean up any previous process
        self._kill_process()

        # command = [
        #     "ffmpeg",
        #     "-y",
        #     "-f", "rawvideo",
        #     "-pix_fmt", "bgr24",
        #     "-s", f"{self._frame_width}x{self._frame_height}",
        #     "-r", str(self._fps),
        #     "-i", "pipe:0",
        #     "-c:v", "h264_nvenc",
        #     # "-c:v", "libx264",
        #     "-preset", "ultrafast",
        #     "-tune", "zerolatency",
        #     "-g", str(int(self._fps * 2)),
        #     "-bufsize", str(10**7),
        #     "-f", "flv",
        #     self._rtmp_url,
        # ]
        command = [
            "/usr/bin/ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._frame_width}x{self._frame_height}",
            "-r", str(self._fps),
            "-i", "-",
            # --- THAY ĐỔI Ở ĐÂY ---
            "-c:v", "h264_omx",        # Dùng OpenMAX thay vì v4l2m2m
            "-b:v", "4000k",           # Bitrate quan trọng với OMX
            "-pix_fmt", "yuv420p", 
            # ---------------------
            "-f", "flv",
            self._rtmp_url,
        ]
        
        print(f"[STREAM] FFmpeg command: {' '.join(command)}")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                bufsize=10**7, # Tăng buffer lên khoảng 10MB
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[STREAM] FFmpeg start failed: {exc}")
            self._process = None
            return False

        time.sleep(0.2)
        if self._process.poll() is not None:
            print("[STREAM] FFmpeg exited early.")
            self._process = None
            return False

        print("[STREAM] FFmpeg started successfully.")
        return True

    def _kill_process(self) -> None:
        """Terminate the current FFmpeg process if any."""
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
        self._enabled = False
        self._kill_process()

    def _restart_ffmpeg(self) -> bool:
        """Try to restart FFmpeg with exponential back-off. Returns True on success."""
        with self._restart_lock:
            for attempt, delay in enumerate(self._RESTART_DELAYS, 1):
                if not self._enabled:
                    return False
                if self._running_checker and not self._running_checker():
                    return False

                print(
                    f"[STREAM] FFmpeg died – restart attempt "
                    f"{attempt}/{len(self._RESTART_DELAYS)} in {delay}s..."
                )
                time.sleep(delay)

                if not self._enabled:
                    return False
                if self._running_checker and not self._running_checker():
                    return False

                if self._spawn_ffmpeg():
                    return True

            print("[STREAM] All restart attempts failed. Streaming disabled.")
            return False

    def _worker(self) -> None:
        while self._running_checker and self._running_checker() and self._enabled:
            # ── get newest frame ──────────────────────────────
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Drop stale frames – always send the newest available
            while not self._queue.empty():
                try:
                    frame = self._queue.get_nowait()
                except queue.Empty:
                    break

            # ── check process health ──────────────────────────
            if self._process is None or self._process.poll() is not None:
                if not self._restart_ffmpeg():
                    break
                continue  # retry with next frame

            # ── write to FFmpeg stdin ─────────────────────────
            try:
                self._process.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                print(f"[STREAM] Write error (pipe broken): {exc}")
                error = self._process.stderr.read().decode()
                print(f"[STREAM] FFmpeg error output: {error}")
                self._kill_process()
                if not self._restart_ffmpeg():
                    break
            except Exception as exc:
                print(f"[STREAM] Write error: {exc}")
