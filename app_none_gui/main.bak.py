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
import datetime
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
from scripts.config import AVATAR_DIR, CAPTURE_ROOT, CHECKIN_DIR, DEFAULT_RTSP_URL, FACE_BUILD_TIME_EXCLUDE, VIDEO, FACE_DATA_DIR, DETECT_SIZE, EMBEDDING_DIM
from scripts.services.attendance import AttendanceService
from scripts.utils.mongodb_access import MongoClientSingleton, now_local, to_local


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
        self.report_interval = 300.0  # Report every 5 minutes
        
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
                
                ok = self.svc.start_capture(self.config.source, self.config.source_type, self.config.stream)
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
# ✅ NEW: Background worker – auto-build face index from videos
# ─────────────────────────────────────────────────────────────

class FaceBuildWorker:
    """
    Periodic background worker that checks the *students* collection for
    documents with ``video_file != ''`` **and** ``has_build_face != true``,
    extracts face embeddings from the referenced video, saves a ``.npy``
    file, rebuilds the Annoy index, and marks the student as processed.

    Parameters
    ----------
    svc : AttendanceService
        The running attendance service (provides ``svc.recognition``).
    interval : float
        Seconds between each check cycle (default 300 = 5 minutes).
    dup_threshold : float
        Cosine-similarity threshold for filtering duplicate embeddings
        extracted from the same video (default 0.7).
    """

    def __init__(
        self,
        svc: "AttendanceService",
        interval: float = 300.0,
        dup_threshold: float = 0.7,
    ) -> None:
        self.svc = svc
        self.interval = interval
        self.dup_threshold = dup_threshold
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log = logging.getLogger("face_build_worker")

    # ── public API ────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="FaceBuildWorker")
        self._thread.start()
        self._log.info("FaceBuildWorker started (interval=%ds)", self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._log.info("FaceBuildWorker stopped.")

    # ── internal loop ─────────────────────────────────────────

    def _loop(self) -> None:
        # Run immediately on start, then every *interval* seconds
        while not self._stop.is_set():
            try:
                # kiểm tra thời gian hiện tại không từ 6:00:00 - 7:00:00, hoặc 13h00 - 14h00
                
                now = now_local().time()
                for start_hour, start_minute, end_hour, end_minute in FACE_BUILD_TIME_EXCLUDE:
                    if datetime.time(start_hour, start_minute) <= now <= datetime.time(end_hour, end_minute):
                        break
                else:
                    self._tick()
            except Exception:
                self._log.exception("Error in FaceBuildWorker tick")
            # Sleep in small increments so we can respond to stop quickly
            slept = 0.0
            while slept < self.interval and not self._stop.is_set():
                time.sleep(min(5.0, self.interval - slept))
                slept += 5.0

    def _tick(self) -> None:
        """One check cycle: query → extract → save → rebuild → update."""
        self._log.info("FaceBuildWorker tick")
        client = MongoClientSingleton.get_client()
        students_col = client.collection("students")
        attendances_col = client.collection("attendances")
        need_rebuild = False
        
        #region video_files
        # Find students that have a video but haven't been processed yet
        query = {
            "video_file": {"$nin": [None, ""]},
            "$or": [
                {"has_build_face": False},
                {"has_build_face": {"$exists": False}},
            ],
        }
        pending = list(students_col.find(query))
        if(len(pending)>0):
            self._log.info("Found %d student(s) to build face index for.", len(pending))

        for doc in pending:
            if self._stop.is_set():
                break
            student_id = doc["_id"]
            student_name = doc.get("name", str(student_id))
            video_path = doc.get("video_file", "")
            video_path = os.path.join(CAPTURE_ROOT, video_path)
            self._log.info(
                "Processing student %s (%s)  video=%s",
                student_name, student_id, video_path,
            )

            if not os.path.isfile(video_path):
                self._log.warning("Video file not found: %s – skipping.", video_path)
                continue

            try:
                embeddings = self._extract_from_video(video_path)
            except Exception:
                self._log.exception("Failed to extract embeddings from %s", video_path)
                continue

            if not embeddings:
                self._log.warning(
                    "No embeddings extracted for %s – skipping.", student_name,
                )
                continue

            # De-duplicate within this video
            unique = self._filter_unique(embeddings)
            self._log.info(
                "  Raw=%d  Unique=%d embeddings", len(embeddings), len(unique),
            )

            # Save .npy (use student _id as filename for safety)
            os.makedirs(FACE_DATA_DIR, exist_ok=True)
            safe_name = str(student_id)
            npy_path = os.path.join(FACE_DATA_DIR, f"{safe_name}.npy")

            # Replace with existing .npy if present
            if os.path.exists(npy_path):
                os.remove(npy_path)

            np.save(npy_path, np.array(unique))
            self._log.info("  Saved %d embeddings → %s", len(unique), npy_path)
            need_rebuild = True

            # Mark student as processed in MongoDB
            try:
                students_col.update_one(
                    {"_id": student_id},
                    {"$set": {"has_build_face": True}},
                )
                self._log.info("  Updated has_build_face=True for %s", student_name)
            except Exception:
                self._log.exception("  Failed to update MongoDB for %s", student_name)
        #endregion
       
        #region avatar_frames
        # Find students that have avatar_frames but haven't been processed yet
        query_avatar = {
            "avatar_frames":  { "$exists": True, "$not": { "$size": 0 } },
            "$or": [
                {"has_build_face": False},
                {"has_build_face": {"$exists": False}},
            ],
        }
        pending_avatar = list(students_col.find(query_avatar))
        if(len(pending_avatar)>0):
            self._log.info("Found %d student(s) with avatar frames to build face index for.", len(pending_avatar))
            
        for doc in pending_avatar:
            if self._stop.is_set():
                break

            student_id = doc["_id"]
            student_name = doc.get("name", str(student_id))


            video_frames_folder = os.path.join(
                AVATAR_DIR,
                str(student_id),
                "frames",
            )

            self._log.info(
                "Processing student %s (%s)  folder=%s",
                student_name, student_id, video_frames_folder,
            )

            if not os.path.isdir(video_frames_folder):
                self._log.warning("Frames folder not found: %s – skipping.", video_frames_folder)
                # Still mark so we don't retry every cycle
                try:
                    students_col.update_one(
                        {"_id": student_id},
                        {"$set": {"has_build_face": True}},
                    )
                except Exception:
                    self._log.exception("  Failed to update student %s", student_id)
                continue

            try:
                embeddings = self._extract_from_folder(video_frames_folder)
            except Exception:
                self._log.exception("Failed to extract embeddings from %s", video_frames_folder)
                continue

            if not embeddings:
                self._log.warning(
                    "No embeddings extracted from %s – skipping.", video_frames_folder,
                )
                try:
                    students_col.update_one(
                        {"_id": student_id},
                        {"$set": {"has_build_face": True}},
                    )
                except Exception:
                    pass
                continue

            # De-duplicate within this folder
            # unique = self._filter_unique(embeddings)
            # self._log.info(
            #     "  Raw=%d  Unique=%d embeddings", len(embeddings), len(unique),
            # )

            # Save / merge into {student_id}.npy
            os.makedirs(FACE_DATA_DIR, exist_ok=True)
            safe_name = str(student_id)
            npy_path = os.path.join(FACE_DATA_DIR, f"{safe_name}.npy")

            if os.path.exists(npy_path):
                os.remove(npy_path)

            np.save(npy_path, np.array(embeddings))
            self._log.info("  Saved %d embeddings → %s", len(embeddings), npy_path)
            need_rebuild = True

            # Mark attendance as processed
            try:
                students_col.update_one(
                    {"_id": student_id},
                    {"$set": {"has_build_face": True}},
                )
                self._log.info("  Updated has_build_face=True for student %s", student_id)
            except Exception:
                self._log.exception("  Failed to update student %s", student_id)
        #endregion
            
        #region attendances
        
        # Tìm attendances đã check mà chưa build
        att_query = {
            "is_checked_image": True,
            "student_id": {"$ne": None},
            "has_build_face": {"$ne": True},
        }
        pending_attendances = list(attendances_col.find(att_query))
        if(len(pending_attendances)>0):
            self._log.info(f'Total attendances to build face: {len(pending_attendances)}')

        if pending_attendances:
            self._log.info(
                "Found %d attendance(s) with checked images to build.",
                len(pending_attendances),
            )

        for doc in pending_attendances:
            if self._stop.is_set():
                break

            attendance_id = doc["_id"]
            student_id = doc.get("student_id")
            student_name = doc.get("student_name", str(student_id))
            timestamp = to_local(doc.get("time"))


            video_frames_folder = os.path.join(
                CHECKIN_DIR,
                timestamp.strftime("%Y-%m-%d"),
                str(attendance_id),
                "frames",
            )

            self._log.info(
                "Processing attendance %s  student=%s (%s)  folder=%s",
                attendance_id, student_name, student_id, video_frames_folder,
            )

            if not os.path.isdir(video_frames_folder):
                self._log.warning("Frames folder not found: %s – skipping.", video_frames_folder)
                # Still mark so we don't retry every cycle
                try:
                    attendances_col.update_one(
                        {"_id": attendance_id},
                        {"$set": {"has_build_face": True}},
                    )
                except Exception:
                    self._log.exception("  Failed to update attendance %s", attendance_id)
                continue

            try:
                embeddings = self._extract_from_folder(video_frames_folder)
            except Exception:
                self._log.exception("Failed to extract embeddings from %s", video_frames_folder)
                continue

            if not embeddings:
                self._log.warning(
                    "No embeddings extracted from %s – skipping.", video_frames_folder,
                )
                try:
                    attendances_col.update_one(
                        {"_id": attendance_id},
                        {"$set": {"has_build_face": True}},
                    )
                except Exception:
                    pass
                continue

            # De-duplicate within this folder
            unique = self._filter_unique(embeddings)
            self._log.info(
                "  Raw=%d  Unique=%d embeddings", len(embeddings), len(unique),
            )

            # Save / merge into {attendance_id}.npy
            os.makedirs(FACE_DATA_DIR, exist_ok=True)
            safe_name = str(attendance_id)
            npy_path = os.path.join(FACE_DATA_DIR, f"{safe_name}.npy")

            if os.path.exists(npy_path):
                os.remove(npy_path)

            np.save(npy_path, np.array(unique))
            self._log.info("  Saved %d embeddings → %s", len(unique), npy_path)
            need_rebuild = True

            # Mark attendance as processed
            try:
                attendances_col.update_one(
                    {"_id": attendance_id},
                    {"$set": {"has_build_face": True}},
                )
                self._log.info("  Updated has_build_face=True for attendance %s", attendance_id)
            except Exception:
                self._log.exception("  Failed to update attendance %s", attendance_id)
        #endregion

        # Rebuild the Annoy index once after processing all students + attendances
        if need_rebuild and not self._stop.is_set():
            self._log.info("Rebuilding Annoy index...")
            try:
                self.svc.recognition.build_face(on_complete=lambda msg: self._log.info(msg))
                self._log.info("Annoy index rebuilt & reloaded successfully.")
            except Exception:
                self._log.exception("Failed to rebuild Annoy index")

    # ── video extraction helpers (reuse svc.recognition model) ──

    def _extract_from_video(self, video_path: str) -> list[np.ndarray]:
        """Read a video file and extract one embedding per detected face per frame."""
        from insightface.utils import face_align as _face_align

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log.error("Cannot open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 100:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log.info("  Video FPS=%.1f  Total frames=%d", fps, total)

        rec = self.svc.recognition  # RecognitionService instance
        all_embs: list[np.ndarray] = []
        count = 0

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            try:
                h, w = frame.shape[:2]
                small = rec.letterbox(frame)
                scale, ox, oy = rec.get_letterbox_params(h, w)
                bboxes, kpss = rec.face_app.models["detection"].detect(small)

                if bboxes.shape[0] == 0:
                    continue
                
                # Find the bbox with the largest area (closest face to the camera)
                max_area = 0
                max_idx = -1
                for i in range(bboxes.shape[0]):
                    x1, y1, x2, y2, score = bboxes[i]
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        max_idx = i

                if max_idx != -1:
                    kps_orig = rec.map_to_original(kpss[max_idx], scale, ox, oy)
                    face_aimg = _face_align.norm_crop(frame, kps_orig)
                    feat = rec.face_app.models["recognition"].get_feat(face_aimg)
                    normed = feat / np.linalg.norm(feat)
                    all_embs.append(normed.flatten())
            except Exception as e:
                self._log.debug("  Frame %d error: %s", count, e)

            if count % 100 == 0:
                self._log.info(
                    "  Processed %d/%d frames  |  Raw embeddings: %d",
                    count, total, len(all_embs),
                )

        cap.release()
        self._log.info(
            "  Done: %d frames processed, %d raw embeddings", count, len(all_embs),
        )
        return all_embs

        
    def _extract_from_folder(self, folder: str) -> list[np.ndarray]:
        """Read all images in *folder* and extract face embeddings."""
        from insightface.utils import face_align as _face_align

        files = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if not files:
            self._log.warning("No images found in %s", folder)
            return []

        self._log.info("  Folder: %s  |  Images: %d", folder, len(files))

        rec = self.svc.recognition
        all_embs: list[np.ndarray] = []

        for i, fname in enumerate(files, 1):
            if self._stop.is_set():
                break

            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                self._log.debug("  Cannot read image: %s", fname)
                continue

            try:
                h, w = img.shape[:2]
                small = rec.letterbox(img)
                scale, ox, oy = rec.get_letterbox_params(h, w)
                bboxes, kpss = rec.face_app.models["detection"].detect(small)

                if bboxes.shape[0] == 0:
                    continue

                for j in range(bboxes.shape[0]):
                    kps_orig = rec.map_to_original(kpss[j], scale, ox, oy)
                    face_aimg = _face_align.norm_crop(img, kps_orig)
                    feat = rec.face_app.models["recognition"].get_feat(face_aimg)
                    normed = feat / np.linalg.norm(feat)
                    all_embs.append(normed.flatten())
            except Exception as e:
                self._log.debug("  Image %s error: %s", fname, e)

            if i % 20 == 0:
                self._log.info(
                    "  Processed %d/%d images  |  Raw embeddings: %d",
                    i, len(files), len(all_embs),
                )

        self._log.info(
            "  Done: %d images processed, %d raw embeddings",
            len(files), len(all_embs),
        )
        return all_embs

    @staticmethod
    def _filter_unique(
        embeddings: list[np.ndarray], threshold: float = 0.8,
    ) -> list[np.ndarray]:
        """Remove near-duplicate embeddings based on cosine similarity."""
        if not embeddings:
            return []
        unique: list[np.ndarray] = [embeddings[0]]
        for emb in embeddings[1:]:
            sims = np.dot(np.array(unique), emb)
            if np.max(sims) < threshold:
                unique.append(emb)
        return unique


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
                if not self.svc.is_running:
                    log.error("Failed to reconnect. Stopping main loop.")
                    break
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
                        # if build_time > target_interval * 0.8:
                        #     log.warning(
                        #         "Frame building slow: %.2fms (target: %.2fms)",
                        #         build_time * 1000, target_interval * 1000
                        #     )
                    except Exception as e:
                        log.error("Error building frame: %s", e)
                        frame = None
                
                # Handle preview display
                if self.config.preview and frame is not None:
                    try:
                        cv2.imshow("Attendance (headless preview)", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("e"):
                            log.info("'e' pressed – stopping.")
                            break
                    except Exception as e:
                        log.error("Preview error: %s", e)
                        break
                else:
                    # ✅ Sleep without preview to avoid busy-waiting
                    time.sleep(0.01)
                
                # Pace to target FPS
                # Kiểm tra source là video file thì check remaining để duy trì target FPS, còn webcam/rtsp thì cứ chạy nhanh nhất có thể
                if self.config.source_type == "Video File":
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

    # try:
    #     log.info("Loading today's attendance records...")
    #     today = svc.load_today_attendance()
    #     log.info("%d attendance records loaded for today.", len(today))
    # except Exception:
    #     log.exception("Failed to load today's attendance (continuing anyway)")

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

    # ── Start face-build background worker ────────────────────
    face_build_worker = FaceBuildWorker(svc, interval=300.0)
    face_build_worker.start()

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
            face_build_worker.stop()
        except Exception:
            log.exception("Error stopping FaceBuildWorker")

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