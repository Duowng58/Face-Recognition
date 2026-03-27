"""
Console / headless attendance application (OPTIMIZED).

Usage examples:
    python -m app_none_gui.main                         # webcam (default)
    python -m app_none_gui.main --source rtsp            # RTSP stream
    python -m app_none_gui.main --source video           # video file from config
    python -m app_none_gui.main --source "rtsp://..."    # custom RTSP URL
    python -m app_none_gui.main --source "C:/path.mp4"   # custom video file
    python -m app_none_gui.main --preview                # show cv2 window

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

# ── Ensure project root is on sys.path ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Logging setup ────────────────────────────────────────────
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"console_{time.strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,  # ✅ Changed from ERROR to INFO for better visibility
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("app_none_gui")

# Import scripts first (loads onnxruntime before any Qt/GUI libs)
from scripts.config import DEFAULT_RTSP_URL, VIDEO
from scripts.services.attendance import AttendanceService


# ═════════════════════════════════════════════════════════════
# ✅ Config dataclass for better maintainability
# ═════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    """Application configuration"""
    source: str | int
    source_type: str
    preview: bool = False
    stream: bool = False
    no_build: bool = False
    
    # ✅ Performance tuning
    target_fps: float = 15.0          # Default target FPS
    frame_timeout: float = 5.0        # Timeout for frame building (seconds)
    reconnect_delay: float = 2.0      # Delay before reconnecting
    max_reconnect_attempts: int = 3   # Max reconnection attempts


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _resolve_source(raw: str) -> tuple[str | int, str]:
    """
    Return ``(source, source_type)`` from the CLI ``--source`` value.

    source_type matches the strings the UI combo-box uses so that
    ``AttendanceService`` behaves identically.
    """
    low = raw.strip().lower()

    if low in ("0", "webcam", "camera"):
        return 0, "Webcam"

    if low in ("rtsp", "rtsp_default"):
        return DEFAULT_RTSP_URL, "RTSP"

    if low in ("video", "video_default"):
        return VIDEO, "Video File"

    # Heuristic: if it starts with rtsp:// treat as RTSP
    if low.startswith("rtsp://"):
        return raw.strip(), "RTSP"

    # Otherwise treat as a video file path
    return raw.strip(), "Video File"


# ─────────────────────────────────────────────────────────────
# Console callbacks
# ─────────────────────────────────────────────────────────────

def _on_status(msg: str) -> None:
    """✅ Optimized: Direct logging without extra formatting"""
    log.info("[STATUS] %s", msg)


def _on_attendance(record) -> None:
    """✅ Optimized: Early return on None values"""
    try:
        if not record:
            return
            
        name = getattr(record, "student_name", None) or "Unknown"
        classroom = getattr(record, "student_classroom", "") or ""
        score = getattr(record, "score", 0.0) or 0.0
        t = getattr(record, "time", None)
        
        # ✅ Faster time formatting
        time_str = t.strftime("%H:%M:%S") if t else "--:--:--"
        
        # ✅ Build log message efficiently (avoid extra concatenations)
        parts = [f"  {name}"]
        if classroom:
            parts.append(f"  |  {classroom}")
        parts.append(f"  |  score={score:.2f}  |  {time_str}")
        
        line = "".join(parts)
        log.info("[ATTENDANCE] %s", line)
        
    except Exception:
        log.exception("Error in _on_attendance callback")


# ─────────────────────────────────────────────────────────────
# ✅ NEW: Performance monitoring
# ─────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """Monitor FPS and frame building time"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.build_times = []
        self.last_report_time = time.time()
        self.report_interval = 10.0  # Report every 10 seconds
        
    def record_frame_time(self, elapsed: float) -> None:
        """Record frame processing time"""
        self.frame_times.append(elapsed)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def record_build_time(self, elapsed: float) -> None:
        """Record frame build time"""
        self.build_times.append(elapsed)
        if len(self.build_times) > self.window_size:
            self.build_times.pop(0)
    
    def should_report(self) -> bool:
        """Check if it's time to report stats"""
        return time.time() - self.last_report_time >= self.report_interval
    
    def report(self) -> None:
        """Log performance statistics"""
        if not self.frame_times:
            return
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        avg_build_time = sum(self.build_times) / len(self.build_times) if self.build_times else 0
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        log.info(
            "PERF: FPS=%.1f (avg frame time: %.2fms, build time: %.2fms)",
            fps, avg_frame_time * 1000, avg_build_time * 1000
        )
        self.last_report_time = time.time()


# ─────────────────────────────────────────────────────────────
# ✅ NEW: Connection manager with retry logic
# ─────────────────────────────────────────────────────────────

class ConnectionManager:
    """Manage capture source connection with automatic retry"""
    
    def __init__(self, config: AppConfig, svc: AttendanceService):
        self.config = config
        self.svc = svc
        self.reconnect_attempts = 0
        
    def connect(self) -> bool:
        """Attempt to connect with retry logic"""
        while self.reconnect_attempts < self.config.max_reconnect_attempts:
            try:
                log.info(
                    "Connecting to %s (attempt %d/%d)...",
                    self.config.source,
                    self.reconnect_attempts + 1,
                    self.config.max_reconnect_attempts
                )
                
                ok = self.svc.start_capture(self.config.source, self.config.source_type)
                if ok:
                    self.reconnect_attempts = 0  # Reset on success
                    log.info("✅ Connected successfully. FPS=%.1f", self.svc.frame_fps)
                    return True
                    
            except Exception as e:
                log.error("Connection failed: %s", e)
            
            self.reconnect_attempts += 1
            if self.reconnect_attempts < self.config.max_reconnect_attempts:
                delay = self.config.reconnect_delay * (2 ** self.reconnect_attempts)  # Exponential backoff
                log.warning("Retrying in %.1f seconds...", delay)
                time.sleep(delay)
        
        log.error("Failed to connect after %d attempts", self.config.max_reconnect_attempts)
        return False


# ─────────────────────────────────────────────────────────────
# ✅ NEW: Main loop refactored for efficiency
# ─────────────────────────────────────────────────────────────

class MainLoop:
    """Manages the main application loop"""
    
    def __init__(self, config: AppConfig, svc: AttendanceService):
        self.config = config
        self.svc = svc
        self.monitor = PerformanceMonitor()
        self.stop_event = threading.Event()
        
    def run(self) -> None:
        """Main loop with optimized frame handling"""
        try:
            import cv2
        except ImportError:
            log.error("OpenCV not installed")
            return
        
        # Calculate target interval
        target_interval = 1.0 / max(self.config.target_fps, 1.0)
        
        log.info("Starting main loop (target FPS: %.1f)", self.config.target_fps)
        
        while not self.stop_event.is_set():
            try:
                t0 = time.perf_counter()
                
                # ✅ OPTIMIZED: Handle preview and streaming efficiently
                frame = None
                if self.config.target_fps != self.svc.frame_fps:
                    self.config.target_fps = self.svc.frame_fps
                    target_interval = 1.0 / max(self.config.target_fps, 1.0)
                    
                # Build frame only if needed
                if self.config.preview or self.config.stream:
                    try:
                        frame = self.svc.build_annotated_frame()
                        build_time = time.perf_counter() - t0
                        self.monitor.record_build_time(build_time)
                        
                        # ✅ Warn if frame building is too slow
                        if build_time > target_interval * 0.8:
                            log.warning(
                                "Frame building slow: %.2fms (target: %.2fms)",
                                build_time * 1000, target_interval * 1000
                            )
                    except Exception as e:
                        log.error("Error building frame: %s", e)
                        frame = None
                
                # Handle preview display
                if self.config.preview and frame is not None:
                    try:
                        cv2.imshow("Attendance (headless preview)", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            log.info("'q' pressed – stopping.")
                            break
                    except Exception as e:
                        log.error("Preview error: %s", e)
                        break
                else:
                    # ✅ Sleep without preview to avoid busy-waiting
                    time.sleep(0.01)
                
                # Pace to target FPS
                elapsed = time.perf_counter() - t0
                remaining = target_interval - elapsed
                
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    log.debug("⚠️ Frame processing exceeded target interval by %.2fms", 
                             (elapsed - target_interval) * 1000)
                
                # Record frame time
                total_time = time.perf_counter() - t0
                self.monitor.record_frame_time(total_time)
                
                # Report performance periodically
                if self.monitor.should_report():
                    self.monitor.report()
                    
            except Exception as e:
                log.exception("Error in main loop iteration")
                time.sleep(0.5)  # Avoid tight error loop
    
    def stop(self) -> None:
        """Signal to stop the main loop"""
        self.stop_event.set()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Console / headless face-recognition attendance system.",
    )
    parser.add_argument(
        "--source", "-s",
        default="0",
        help=(
            "Video source: 0/webcam, rtsp, video, "
            "or a direct URL / file path. Default: 0 (webcam)."
        ),
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Show a live OpenCV window (requires a display).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable RTMP streaming on start.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        dest="no_build",
        help="Skip rebuilding the Annoy face index on start.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target FPS (default: 15.0).",
    )
    args = parser.parse_args()

    source, source_type = _resolve_source(args.source)
    
    # ✅ Create config object
    config = AppConfig(
        source=source,
        source_type=source_type,
        preview=args.preview,
        stream=args.stream,
        no_build=args.no_build,
        target_fps=args.fps,
    )

    log.info("=" * 60)
    log.info("Console attendance app starting")
    log.info("Source: %s  Type: %s", config.source, config.source_type)
    log.info("Preview: %s  Stream: %s  FPS: %.1f", config.preview, config.stream, config.target_fps)
    log.info("Log file: %s", LOG_FILE)
    log.info("=" * 60)

    # ── Create service ────────────────────────────────────────
    try:
        svc = AttendanceService()
    except Exception:
        log.exception("Failed to create AttendanceService")
        sys.exit(1)

    # Wire callbacks
    svc.on_status = _on_status
    svc.on_attendance = _on_attendance

    # ✅ NEW: Create main loop with config
    main_loop = MainLoop(config, svc)

    def _on_video_end() -> None:
        log.info("Video file ended.")
        main_loop.stop()

    svc.on_video_end = _on_video_end

    # Wire tracker disappeared callback
    svc.tracker.on_disappeared = svc.handle_disappeared

    # ── Graceful shutdown on Ctrl+C / SIGTERM ─────────────────
    def _signal_handler(sig, frame):
        log.info("Caught signal %s, shutting down...", sig)
        main_loop.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Load recognition assets ───────────────────────────────
    try:
        log.info("Loading recognition model & index...")
        svc.load_recognition_assets()
    except Exception:
        log.exception("Failed to load recognition assets")
        svc.close()
        sys.exit(1)

    try:
        log.info("Loading classroom data...")
        svc.refresh_classrooms()
    except Exception:
        log.exception("Failed to load classrooms (continuing anyway)")

    try:
        log.info("Loading today's attendance records...")
        today = svc.load_today_attendance()
        log.info("%d attendance records loaded for today.", len(today))
    except Exception:
        log.exception("Failed to load today's attendance (continuing anyway)")

    # ── Start capture with connection manager ──────────────────
    conn_manager = ConnectionManager(config, svc)
    if not conn_manager.connect():
        log.error("Failed to establish connection")
        svc.close()
        sys.exit(1)

    if args.stream:
        svc.toggle_streaming(True)
        log.info("RTMP streaming enabled.")

    if args.preview:
        log.info("Preview window enabled. Press 'q' in the window to quit.")

    log.info("Running... Press Ctrl+C to stop.\n")

    # ── Run main loop ─────────────────────────────────────────
    try:
        main_loop.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – stopping.")
    except Exception:
        log.exception("Unhandled exception in main loop")
    finally:
        # ── Cleanup ───────────────────────────────────────────
        log.info("Shutting down...")
        try:
            svc.close()
        except Exception:
            log.exception("Error during service close")

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logging.getLogger("app_none_gui").exception("Unhandled exception in main()")
        sys.exit(1)