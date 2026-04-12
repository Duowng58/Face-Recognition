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
        queue_size: int = 1,
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
        
        self._is_process_spawning = False
        self._retry_count = 0

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def toggle(self, enabled: bool, fps: float, is_running: Callable[[], bool]) -> None:
        """Hàm công tắc được gọi từ attendance.py"""
        self._enabled = enabled
        if not enabled:
            self.stop()
            return
        # Nếu đang chạy chính và chưa có luồng stream thì khởi tạo
        if is_running() and not self._is_process_running():
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

    def _is_process_running(self) -> bool:
        check = (self._gst_process is not None and self._gst_process.poll() is None) and \
               (self._ffmpeg_process is not None and self._ffmpeg_process.poll() is None)
        return check

    def _spawn_hybrid(self) -> bool:
        if self._is_process_spawning:
            return False
        
        self._is_process_spawning = True
        self._kill_process()
        print(f'[STREAM] Spawning hybrid for {self._rtmp_url}...')

        # Tính toán kích thước buffer cho 1 frame YUV420
        # YUV420 tốn 1.5 byte mỗi pixel
        frame_size = int(self._frame_width * self._frame_height * 1.5)

        # Lệnh GStreamer tích hợp thẳng FFmpeg qua ống dẫn nội bộ
        # Dùng 'omxh264enc' cho Jetson (tăng tốc phần cứng)
        # Kết nối thẳng tới rtmpsink (hoặc dùng flvmux ! fdsink)
        gst_cmd = [
            "gst-launch-1.0", "-q",
            "fdsrc", f"blocksize={frame_size}", "!",
            "videoparse", "format=2", f"width={self._frame_width}", f"height={self._frame_height}", "!",
            "omxh264enc", 
            "bitrate=4000000", "control-rate=2", "iframeinterval=20", "preset-level=1", "!",
            "h264parse", "!",
            "flvmux", "streamable=true", "!",
            "fdsink"
        ]

        # FFmpeg chỉ đóng vai trò nhận stream đã đóng gói FLV và đẩy lên RTMP (Cực nhẹ CPU)
        # Thêm các flag tối ưu cho FFmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",            # Chỉ hiện lỗi để log gọn sạch
            "-i", "pipe:0", 
            "-c", "copy", 
            "-f", "flv", 
            # "-flvflags", "no_duration_filesize", # Tránh lỗi metadata khi stream
            # "-rtmp_tcppayload_size", "1312",     # Tối ưu kích thước gói tin mạng
            self._rtmp_url
        ]

        try:
            # Tạo process GStreamer
            self._gst_process = subprocess.Popen(
                gst_cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, # Chuyển output sang cho FFmpeg
                stderr=subprocess.DEVNULL,
                bufsize=0 # Đẩy dữ liệu real-time
            )
            
            # Tạo process FFmpeg nhận đầu vào từ stdout của GStreamer
            self._ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=self._gst_process.stdout, 
                stderr=subprocess.DEVNULL,
                bufsize=0 # Đẩy dữ liệu real-time
            )
            
            time.sleep(0.5)
            return self._is_process_running()
        except Exception as exc:
            print(f"[STREAM] Spawn error: {exc}")
            return False
        finally:
            self._is_process_spawning = False

    def enqueue(self, frame) -> None:
        """Đẩy frame vào hàng đợi"""
        if not self._enabled:
            return
        
        # Ép buộc giải phóng chỗ trống nếu đầy
        # Dùng while thay vì if để chắc chắn queue có chỗ (phòng hờ thread khác can thiệp)
        while self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break # Queue đã trống, thoát vòng lặp ngay
            
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            print('[STREAM] Queue is full, dropping frame')
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
                
                try:
                    proc.kill()
                except: pass
        self._gst_process = None
        self._ffmpeg_process = None

    def _worker(self) -> None:
        print("[STREAM] Worker started")
        while self._running_checker and self._running_checker() and self._enabled:
            try:
                # 1. Lấy frame (đợi tối đa 0.5s nếu queue trống)
                # Vì enqueue đã đảm bảo frame trong queue là mới nhất, 
                # ta chỉ cần lấy 1 cái duy nhất ở đây.
                frame = self._queue.get(timeout=0.5)
                
            except queue.Empty:
                print("[STREAM] Queue is empty, waiting for frame...")
                continue

            # Kiểm tra trạng thái process
            if not self._is_process_running():
                # Tính toán delay retry (nếu cần) và khởi động lại
                delay = self._RESTART_DELAYS[min(self._retry_count, len(self._RESTART_DELAYS) - 1)]
                print(f"[STREAM] Process not running, retrying in {delay} seconds...")
                time.sleep(delay)
                if self._spawn_hybrid():
                    self._retry_count = 0
                else:
                    self._retry_count += 1
                    continue

            try:
                # Chuyển đổi sang YUV420P - OpenCV làm việc này cực tốt trên Jetson
                yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                data = yuv_frame.tobytes()

                # Đẩy dữ liệu vào stdin của GStreamer
                if self._gst_process and self._gst_process.stdin:
                    self._gst_process.stdin.write(data)
                    self._gst_process.stdin.flush()
            except (BrokenPipeError, IOError):
                print("[STREAM] Pipe broken, restarting...")
                self._kill_process()
            except Exception as e:
                print(f"[STREAM] Write error: {e}")
        
        self._kill_process()
        print('[STREAM] Worker stopped')
