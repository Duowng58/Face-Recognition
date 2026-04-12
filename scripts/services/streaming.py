import queue
import subprocess
import threading
import time
from typing import Callable, Optional
import cv2

class StreamingService:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        rtmp_url: str,
        queue_size: int = 2, # Tăng lên 2 để mượt hơn trên Mac
    ) -> None:
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._rtmp_url = rtmp_url
        self._queue = queue.Queue(maxsize=queue_size)
        
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._running_checker: Optional[Callable[[], bool]] = None
        self._fps: float = 25.0
        self._is_spawning = False

    def _is_process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _spawn_ffmpeg(self) -> bool:
        if self._is_spawning:
            return False
        
        self._is_spawning = True
        self._kill_process()
        
        print(f'[STREAM] Mac M-Series Optimized: Streaming to {self._rtmp_url}')

        # Lệnh FFmpeg sử dụng VideoToolbox (Hardware Acceleration cho Mac)
        command = [
            '/opt/homebrew/bin/ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{self._frame_width}x{self._frame_height}",
            '-r', str(int(self._fps)),
            '-i', '-', 
            '-c:v', 'h264_videotoolbox',
            '-b:v', '4000k',
            '-maxrate', '4000k',            # Khống chế bitrate tối đa
            '-bufsize', '8000k',            # Buffer giúp mượt hình khi mạng trồi sụt
            '-pix_fmt', 'yuv420p',
            '-g', str(int(self._fps * 2)),  # Keyframe mỗi 2 giây (Cần thiết cho RTMP)
            '-profile:v', 'main',
            '-realtime', '1',
            '-flvflags', 'no_duration_filesize',
            '-f', 'flv',
            self._rtmp_url
        ]

        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, # Đổi thành subprocess.PIPE nếu muốn debug lỗi FFmpeg
                bufsize=0
            )
            time.sleep(0.5) # Đợi FFmpeg khởi tạo pipe
            return self._is_process_running()
        except Exception as e:
            print(f"[STREAM] Spawn error: {e}")
            return False
        finally:
            self._is_spawning = False

    def _worker(self) -> None:
        print("[STREAM] Worker started (FFmpeg Native)")
        while self._running_checker and self._running_checker() and self._enabled:
            try:
                # Lấy frame mới nhất từ hàng đợi
                frame = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not self._is_process_running():
                print("[STREAM] FFmpeg process died, restarting...")
                if self._spawn_ffmpeg():
                    time.sleep(1)
                continue

            try:
                if self._process and self._process.stdin:
                    # Ghi trực tiếp bytes của frame vào stdin của FFmpeg
                    self._process.stdin.write(frame.tobytes())
                    self._process.stdin.flush()
            except (BrokenPipeError, IOError):
                self._kill_process()
            except Exception as e:
                print(f"[STREAM] Write error: {e}")
        
        self._kill_process()
        print("[STREAM] Worker stopped")

    def enqueue(self, frame) -> None:
        """Hàm này được gọi từ vòng lặp camera để đẩy frame vào"""
        if not self._enabled:
            return
        
        # Nếu queue đầy, xóa frame cũ nhất để đảm bảo không bị delay (latency)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def start(self, fps: float, is_running: Callable[[], bool]) -> None:
        if not self._enabled or not self._rtmp_url:
            return
        self._fps = fps
        self._running_checker = is_running
        
        if self._spawn_ffmpeg():
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def stop(self) -> None:
        self._enabled = False
        self._kill_process()

    def _kill_process(self) -> None:
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=0.5)
            except:
                pass
            self._process = None

    def toggle(self, enabled: bool, fps: float, is_running: Callable[[], bool]) -> None:
        """Hàm công tắc tương thích với attendance.py"""
        self._enabled = enabled
        if not enabled:
            self.stop()
        elif is_running() and not self._is_process_running():
            self.start(fps, is_running)
