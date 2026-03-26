"""
Console / headless attendance application.

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
import traceback

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
    level=logging.ERROR,
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


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_source(raw: str) -> tuple:
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
    log.info("[STATUS] %s", msg)


def _on_attendance(record) -> None:
    try:
        name = record.student_name or "Unknown"
        classroom = getattr(record, "student_classroom", "") or ""
        score = getattr(record, "score", 0) or 0
        t = getattr(record, "time", None)
        time_str = t.strftime("%H:%M:%S") if t else "--:--:--"
        line = f"  {name}"
        if classroom:
            line += f"  |  {classroom}"
        line += f"  |  score={score:.2f}  |  {time_str}"
        log.info("[ATTENDANCE] %s", line)
    except Exception:
        log.exception("Error in _on_attendance callback")


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
    args = parser.parse_args()

    source, source_type = _resolve_source(args.source)

    log.info("=" * 60)
    log.info("Console attendance app starting")
    log.info("Source: %s  Type: %s", source, source_type)
    log.info("Preview: %s  Stream: %s", args.preview, args.stream)
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

    stop_event = threading.Event()

    def _on_video_end() -> None:
        log.info("Video file ended.")
        stop_event.set()

    svc.on_video_end = _on_video_end

    # Wire tracker disappeared callback
    svc.tracker.on_disappeared = svc.handle_disappeared

    # ── Graceful shutdown on Ctrl+C / SIGTERM ─────────────────
    def _signal_handler(sig, frame):
        log.info("Caught signal %s, shutting down...", sig)
        stop_event.set()

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

    # if not args.no_build:
    #     log.info("Building face index...")
    #     svc.build_face()

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

    # ── Start capture ─────────────────────────────────────────
    log.info("Starting capture: source=%s  type=%s", source, source_type)
    try:
        ok = svc.start_capture(source, source_type)
    except Exception:
        log.exception("Exception while opening capture source")
        svc.close()
        sys.exit(1)

    if not ok:
        log.error("Cannot open source: %s", source)
        svc.close()
        sys.exit(1)

    log.info("Capture started.  FPS=%.1f", svc.frame_fps)

    if args.stream:
        svc.toggle_streaming(True)
        log.info("RTMP streaming enabled.")

    if args.preview:
        log.info("Preview window enabled. Press 'q' in the window to quit.")

    log.info("Running... Press Ctrl+C to stop.\n")

    # ── Main loop ─────────────────────────────────────────────
    try:
        import cv2

        while not stop_event.is_set():
            try:
                if args.preview:
                    frame = svc.build_annotated_frame()
                    if frame is not None:
                        cv2.imshow("Attendance (headless preview)", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            log.info("'q' pressed – stopping.")
                            break
                    else:
                        time.sleep(0.02)
                elif args.stream:
                    # No preview but streaming – must build frames to feed RTMP
                    frame = svc.build_annotated_frame()
                    if frame is None:
                        time.sleep(0.02)
                else:
                    # No preview, no streaming – just keep the main thread alive
                    stop_event.wait(timeout=1.0)
            except Exception:
                log.exception("Error in main loop iteration")
                time.sleep(0.5)  # avoid tight error loop

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – stopping.")

    # ── Cleanup ───────────────────────────────────────────────
    log.info("Shutting down...")
    try:
        svc.close()
    except Exception:
        log.exception("Error during service close")

    if args.preview:
        try:
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
