import queue
import subprocess
import threading
import time
from typing import Callable, Optional
import numpy as np
import cv2
class StreamingService:
    _RESTART_DELAYS = (1, 2, 5, 10, 15, 30)

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
        
        self._gst_process: Optional[subprocess.Popen] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        
        self._thread: Optional[threading.Thread] = None
        self._enabled = True
        self._running_checker: Optional[Callable[[], bool]] = None
        self._fps: float = 25.0
        self._restart_lock = threading.Lock()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def toggle(self, enabled: bool, fps: float, is_running: Callable[[], bool]) -> None:
        """Hàm công tắc được gọi từ attendance.py"""
        self._enabled = enabled
        if not enabled:
            self.stop()
            return
        # Nếu đang chạy chính và chưa có luồng stream thì khởi tạo
        if is_running() and (self._gst_process is None or self._gst_process.poll() is not None):
            self.start(fps, is_running)

    def start(self, fps: float, is_running: Callable[[], bool]) -> None:
        if not self._enabled or not self._rtmp_url:
            return
        self._fps = fps
        self._running_checker = is_running
        if not self._spawn_hybrid():
            return
        # Chỉ tạo thread worker nếu chưa có hoặc thread cũ đã chết
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _spawn_hybrid(self) -> bool:
        self._kill_process()
        
        # GStreamer: Nén phần cứng OMX (BGR -> H264)
        gst_cmd = [
            "gst-launch-1.0", "-q",
            "fdsrc", "!",
            "videoparse", "format=2", f"width={self._frame_width}", f"height={self._frame_height}", "!",
            "omxh264enc", 
            "bitrate=4500000",      # Tăng lên 4.5M để ảnh nét hơn
            "control-rate=2",       # Chế độ CBR (Constant Bitrate) ổn định cho livestream
            "iframeinterval=50",    # Cứ 2 giây tạo 1 Keyframe (nếu FPS=25) giúp hồi phục ảnh nhanh
            "preset-level=1",       # Tối ưu tốc độ xử lý cho chip phần cứng
            "!",
            "h264parse", "!",
            "flvmux", "streamable=true", "!",
            "fdsink"
        ]

        # FFmpeg: Copy luồng đã nén lên RTMP (CPU ~ 0%)
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-c", "copy", "-f", "flv", self._rtmp_url]

        try:
            self._gst_process = subprocess.Popen(gst_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=self._gst_process.stdout, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            return self._gst_process.poll() is None
        except Exception as exc:
            print(f"[STREAM] Start error: {exc}")
            return False

    def enqueue(self, frame) -> None:
        """Đẩy frame vào hàng đợi"""
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._enabled = False
        self._kill_process()

    def _kill_process(self) -> None:
        for proc in [self._ffmpeg_process, self._gst_process]:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.2)
                except: pass
        self._gst_process = None
        self._ffmpeg_process = None

    def _worker(self) -> None:
        while self._running_checker and self._running_checker() and self._enabled:
            try:
                frame = self._queue.get(timeout=0.5)
                while not self._queue.empty(): frame = self._queue.get_nowait()
            except queue.Empty: continue

            if self._gst_process is None or self._gst_process.poll() is not None:
                self._spawn_hybrid()
                continue

            try:
                # 1. Chuyển đổi BGR sang YUV420P (I420) ngay trong Python
                # OpenCV chuyển đổi này cực nhanh trên Jetson
                yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                
                # 2. Đảm bảo dữ liệu phẳng và liên tục
                data = yuv_frame.tobytes()
                self._gst_process.stdin.write(data)
                self._gst_process.stdin.flush()
            except:
                self._kill_process()
