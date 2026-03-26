"""
Framework-agnostic attendance service.

Encapsulates: video capture, face detection, tracker-embedding matching,
attendance registration, frame annotation, and save-queue management.

Any UI (PySide6, Tkinter, headless …) can instantiate ``AttendanceService``
and wire its callbacks to UI updates.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from bson import ObjectId

from scripts.config import (
    ANNOY_INDEX_PATH,
    AVATAR_DIR,
    CHECKIN_DIR,
    DEFAULT_RTMP_URL,
    EMBEDDING_DIM,
    FACE_DATA_DIR,
    FONT_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAPPING_PATH,
    MIN_BBOX_AREA,
    SIM_THRESHOLD,
    TREE,
    VIDEO_FOURCC,
    VIDEO_FPS,
)
from scripts.services.face_tracker import Tracker
from scripts.services.recognition import RecognitionService
from scripts.services.streaming import StreamingService
from scripts.utils.cv2_helper import check_blur_laplacian, cv2_putText_utf8
from scripts.utils.image_utils import get_attendance_frame_path, get_student_avatar_path
from scripts.utils.mongodb_access import (
    Attendance,
    AttendanceRepository,
    MongoClientSingleton,
    Student,
    StudentRepository,
    now_local,
    start_of_today_local,
    to_local,
)


# ──────────────────────────────────────────────────────────────
# Tiny helpers
# ──────────────────────────────────────────────────────────────

def compute_iou(bb1, bb2) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    a1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    a2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / float(a1 + a2 - inter)


def crop_face_padded(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad: float = 0.2) -> np.ndarray:
    """Crop a face region with *pad* relative padding, clamped to frame bounds."""
    dy = int(abs(y2 - y1) * pad)
    dx = int(abs(x2 - x1) * pad)
    return frame[
        max(y1 - dy, 0): max(y2 + dy, 0),
        max(x1 - dx, 0): max(x2 + dx, 0),
    ]


def annotate_frame(
    frame: np.ndarray,
    rects: list,
    stats: list[tuple[str, int]] | None = None,
) -> np.ndarray:
    """Draw bounding boxes, labels, and optional stats overlay on *frame* (mutates in-place)."""
    for track_id, color, label, (x1, y1, x2, y2), detect_score in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if os.path.exists(FONT_PATH):
            frame = cv2_putText_utf8(frame, label + ' - ' + f"{detect_score:.2f}", (x1, y1 - 40), FONT_PATH, 30, color)
        else:
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if stats:
        for text, y in stats:
            cv2.putText(frame, text, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame


# ──────────────────────────────────────────────────────────────
# AttendanceService
# ──────────────────────────────────────────────────────────────

class AttendanceService:
    """
    Manages capture → detect → track → recognise → register pipeline.

    **Callbacks** (set them before calling ``start_capture``):

    * ``on_status(msg: str)``            – status bar text
    * ``on_attendance(record: Attendance)`` – new attendance row
    * ``on_frame_ready(annotated_bgr: np.ndarray)`` – frame ready to render
    """

    def __init__(self) -> None:
        # ── services ──────────────────────────────
        self.recognition = RecognitionService(
            face_data_dir=FACE_DATA_DIR,
            annoy_index_path=ANNOY_INDEX_PATH,
            mapping_path=MAPPING_PATH,
            embedding_dim=EMBEDDING_DIM,
            tree=TREE,
            sim_threshold=SIM_THRESHOLD,
        )
        self.streaming = StreamingService(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            rtmp_url=DEFAULT_RTMP_URL,
        )
        self.tracker = Tracker()

        # ── db ────────────────────────────────────
        self.attendance_repo = AttendanceRepository()
        self.student_repo = StudentRepository()

        # ── state ─────────────────────────────────
        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._source: int | str = 0
        self._source_type: str = "Webcam"
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_faces: list = []
        self._capture_thread: Optional[threading.Thread] = None
        self._detect_thread: Optional[threading.Thread] = None

        self._trackid_to_name: Dict[int, dict] = {}
        self._last_seen: Dict[str, float] = {}

        self.current_detect_time: float = 0.0
        self.current_faces: int = 0
        self.current_faces_valid: int = 0
        self.frame_count: int = 0
        self.detect_frame_count: int = 0
        self._frame_fps: float = 25.0

        self.list_classrooms: list = []

        # ── save queue ────────────────────────────
        self._save_queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._save_worker, daemon=True).start()

        os.makedirs(CHECKIN_DIR, exist_ok=True)

        # ── callbacks (UI sets these) ─────────────
        self.on_status: Callable[[str], None] = lambda msg: None
        self.on_attendance: Callable[[Attendance], None] = lambda rec: None
        # called from the detect thread – UI should schedule on main thread
        self.on_frame_ready: Callable[[np.ndarray, list], None] = lambda f, r: None
        self.on_video_end: Callable[[], None] = lambda: None

    # ------------------------------------------------------------------
    # Recognition helpers
    # ------------------------------------------------------------------

    def load_recognition_assets(self) -> None:
        """Load face model + Annoy index in a background thread."""
        self.on_status("Thông báo: Đang tải mô hình nhận diện...")
        self.recognition.load_assets(self.on_status)

    def build_face(self) -> None:
        """Rebuild the Annoy index from *.npy files."""
        self.recognition.build_face(on_complete=self.on_status)

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        return self.recognition.recognize(embedding)

    # ------------------------------------------------------------------
    # Classroom data
    # ------------------------------------------------------------------

    def refresh_classrooms(self) -> None:
        client = MongoClientSingleton.get_client()
        self.list_classrooms = list(client.db["classrooms"].find())

    def load_today_attendance(self) -> List[Attendance]:
        return self.attendance_repo.find({"time": {"$gte": start_of_today_local()}})

    # ------------------------------------------------------------------
    # Capture control
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_fps(self) -> float:
        return self._frame_fps

    def start_capture(self, source: int | str, source_type: str) -> bool:
        """
        Open *source* and start capture + detect threads.

        Returns ``True`` on success, ``False`` if the source cannot be opened.
        """
        if self._running:
            return True

        if source == 0:
            try:
                temp = cv2.VideoCapture(0)
                if temp.isOpened():
                    temp.release()
            except Exception as e:
                print(f"Error opening camera 0: {e}")

        backend = cv2.CAP_ANY if source == 0 else cv2.CAP_FFMPEG
        self._source = source
        self._source_type = source_type
        self._capture = cv2.VideoCapture(source, backend)
        if source != 0:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            return False

        w = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        print(w, h, fps)

        self._frame_fps = fps if 0 < fps <= 100 else 25.0
        self._running = True
        self._latest_faces = []

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._detect_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._detect_thread.start()

        self.streaming.start(self._frame_fps, lambda: self._running)
        return True

    def stop_capture(self) -> None:
        if not self._running and self._capture is None:
            return

        self._running = False

        for t in (self._capture_thread, self._detect_thread):
            if t is not None and t.is_alive():
                t.join(timeout=1)

        self.streaming.stop()
        self._release_capture()

    def toggle_streaming(self, enabled: bool) -> None:
        self.streaming.toggle(enabled, self._frame_fps, lambda: self._running)

    # ------------------------------------------------------------------
    # Internal: capture loop
    # ------------------------------------------------------------------

    def _release_capture(self) -> None:
        if self._capture is None:
            return
        try:
            if self._capture.isOpened():
                self._capture.release()
        except cv2.error:
            pass
        finally:
            self._capture = None

        try:
            if hasattr(self.tracker, "release"):
                self.tracker.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auto-reconnect
    # ------------------------------------------------------------------

    _RECONNECT_DELAYS = (1, 2, 5, 10, 15, 30)  # seconds – exponential back-off

    def _reconnect_capture(self) -> bool:
        """
        Try to re-open ``self._source`` with exponential back-off.

        Returns ``True`` if reconnected successfully, ``False`` if all
        attempts failed or ``self._running`` was set to ``False`` externally.
        """
        # Release the broken capture first
        try:
            if self._capture is not None and self._capture.isOpened():
                self._capture.release()
        except cv2.error:
            pass
        self._capture = None

        for attempt, delay in enumerate(self._RECONNECT_DELAYS, 1):
            if not self._running:
                return False

            self.on_status(
                f"Thông báo: Mất kết nối nguồn video – thử kết nối lại "
                f"lần {attempt}/{len(self._RECONNECT_DELAYS)} sau {delay}s..."
            )
            print(
                f"[RECONNECT] Attempt {attempt}/{len(self._RECONNECT_DELAYS)} "
                f"in {delay}s  (source={self._source})"
            )
            time.sleep(delay)

            if not self._running:
                return False

            backend = cv2.CAP_ANY if self._source == 0 else cv2.CAP_FFMPEG
            # cv2.CAP_GSTREAMER
            cap = cv2.VideoCapture(self._source, backend)
            if self._source != 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                self._capture = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                self._frame_fps = fps if 0 < fps <= 100 else 25.0
                self.on_status("Thông báo: Đã kết nối lại nguồn video thành công!")
                print("[RECONNECT] Success")
                return True

            try:
                cap.release()
            except cv2.error:
                pass

        self.on_status("Thông báo: Không thể kết nối lại nguồn video.")
        print("[RECONNECT] All attempts failed")
        return False

    def _capture_loop(self) -> None:
        self.frame_count = 0
        consecutive_failures = 0
        _MAX_CONSECUTIVE_FAILURES = 30  # ~0.3s of consecutive read() failures

        while self._running:
            if self._capture is None or not self._capture.isOpened():
                # Source lost – attempt reconnect (skip for video files)
                if self._source_type == "Video File":
                    self.on_video_end()
                    break
                if not self._reconnect_capture():
                    break  # all retries exhausted
                consecutive_failures = 0
                continue

            try:
                ret, frame = self._capture.read()
                self.frame_count += 1
                if self.frame_count > 1000:
                    self.frame_count = 0
            except cv2.error as e:
                print(f"[CAPTURE] cv2 error: {e}")
                if self._source_type == "Video File":
                    self.on_video_end()
                    break
                # Trigger reconnect on next iteration
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
                continue

            if not ret:
                if self._source_type == "Video File":
                    self.on_video_end()
                    break
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"[CAPTURE] {consecutive_failures} consecutive read failures – reconnecting")
                    try:
                        self._capture.release()
                    except Exception:
                        pass
                    self._capture = None
                    consecutive_failures = 0
                else:
                    time.sleep(0.01)
                continue

            consecutive_failures = 0
            with self._frame_lock:
                self._latest_frame = frame
            if self._source_type == "Video File":
                time.sleep(1 / self._frame_fps)

        self._running = False

    # ------------------------------------------------------------------
    # Internal: detection worker
    # ------------------------------------------------------------------

    def _detect_worker(self) -> None:
        self.detect_frame_count = 0
        last_frame_count = None

        while self._running:
            with self._frame_lock:
                if self._latest_frame is None:
                    time.sleep(1 / self._frame_fps)
                    continue
                frame_copy = self._latest_frame.copy()
                frame_count = self.frame_count

            if last_frame_count == frame_count:
                time.sleep(1 / self._frame_fps)
                continue
            last_frame_count = frame_count

            self.detect_frame_count += 1
            if self.detect_frame_count > 1000:
                self.detect_frame_count = 0

            # ── face detection ──
            try:
                t0 = time.time()
                # faces = self.recognition.face_app.get(frame_copy)
                faces = self.recognition.get_embeddings(frame_copy)
                self.current_detect_time = time.time() - t0
                self.current_faces = len(faces)
            except Exception as e:
                print(f"Error during face detection: {e}")
                continue

            rects: list = []
            embeddings: list = []
            confidences: list = []

            for face in faces:
                # print(face)
                x1, y1, x2, y2 = face['bbox']
                area = (x2 - x1) * (y2 - y1)
                # if area < MIN_BBOX_AREA:
                #     continue

                # face_crop = crop_face_padded(frame_copy, x1, y1, x2, y2)
                # mask_label, mask_conf = self.recognition.check_mask(face_crop)
                # if mask_label == "Mask":
                #     continue

                rects.append([x1, y1, x2, y2])
                embeddings.append(face['normed_embedding'])
                confidences.append(face['det_score'])

            self.current_faces_valid = len(embeddings)

            # ── tracking ──
            tracks = self.tracker.update(rects, classId="face")
            rects2: list = []
            used_rect_indices: Set[int] = set()

            for track in tracks:
                x1, y1, x2, y2, track_id, _ = track

                matched_embedding = None
                best_iou = 0.0
                best_rect_idx = -1
                for i, (rect, emb, conf) in enumerate(zip(rects, embeddings, confidences)):
                    if i in used_rect_indices:
                        continue
                    iou_val = compute_iou([x1, y1, x2, y2], rect)
                    if iou_val > 0.5 and iou_val > best_iou:
                        best_iou = iou_val
                        best_rect_idx = i
                        matched_embedding = (emb, rect, conf)

                if matched_embedding is None:
                    continue
                used_rect_indices.add(best_rect_idx)

                # ── init tracker entry ──
                if self._trackid_to_name.get(track_id) is None:
                    self._trackid_to_name[track_id] = {
                        "name": "Unknown",
                        "score": 0.0,
                        "student": None,
                        "frames": [],
                        "frame": None,
                        "attendance_id": None,
                    }

                tracker = self._trackid_to_name[track_id]
                name, score = tracker["name"], tracker["score"]
                tracker_frame_index = tracker.get("frame_index", 0)

                if tracker["name"] == "Unknown" or (tracker_frame_index + 5) < frame_count or tracker_frame_index > frame_count:
                    new_name, new_score = self.recognize(matched_embedding[0])
                    tracker["frame_index"] = frame_count
                    if tracker["name"] == "Unknown":
                        tracker["name"] = new_name
                        tracker["score"] = new_score
                    elif new_name != "Unknown" and new_name != tracker["name"]:
                        if new_score > (tracker["score"] + 0.05):
                            tracker["name"] = new_name
                            tracker["score"] = new_score
                            tracker["frames"] = []

                name, score = tracker["name"], tracker["score"]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = (
                    f"ID:{track_id} - {score:.2f} - {matched_embedding[2]:.2f}"
                    if name != "Unknown"
                    else f"ID:{track_id} Unknown"
                )

                # ── collect face crops for the tracker ──
                if frame_copy is not None:
                    face_crop_avatar = crop_face_padded(frame_copy, x1, y1, x2, y2)
                    # is_blur, variance = check_blur_laplacian(face_crop_avatar)
                    is_blur = False  # currently disabled
                    detect_score = matched_embedding[2]

                    if not is_blur:
                        if not any(self.detect_frame_count is c for _, c, f, e in tracker["frames"]):
                            tracker["frames"].append((detect_score, self.detect_frame_count, face_crop_avatar, matched_embedding[0]))

                        if detect_score > tracker.get("variance", 0):
                            tracker["frame"] = face_crop_avatar.copy()
                            tracker["variance"] = detect_score

                tracker["frames"] = sorted(tracker["frames"], key=lambda x: x[0], reverse=True)[:15]
                rects2.append((track_id, color, label, [x1, y1, x2, y2], detect_score))

            self._latest_faces = rects2, frame_copy

    # ------------------------------------------------------------------
    # Tracker disappeared → attendance logic
    # ------------------------------------------------------------------

    def handle_disappeared(self, track_id: int) -> None:
        """Called when the tracker drops a track. Runs attendance logic in bg thread."""
        print(f"Object {track_id} disappeared")
        tracker = self._trackid_to_name.get(track_id)
        if tracker is None:
            return
        print(
            f"Track ID: {track_id}, total frames: {len(tracker['frames'])}, "
            f"name: {tracker.get('name', '--')}, variance: {tracker.get('variance', 0)}, "
            f"score: {tracker.get('score', 0)}, student: {tracker.get('student', '--')}"
        )

        snapshot = {
            "frames": list(tracker.get("frames", [])),
            "frame": tracker.get("frame"),
            "name": tracker.get("name"),
            "score": tracker.get("score", 0),
            "attendance_id": tracker.get("attendance_id"),
            "video_writer": tracker.get("video_writers"),
        }
        self._trackid_to_name.pop(track_id, None)

        threading.Thread(
            target=self._process_disappeared,
            args=(track_id, snapshot),
            daemon=True,
        ).start()

    def _process_disappeared(self, track_id: int, snapshot: dict) -> None:
        timestamp = now_local()

        def save_frames(attendance_id: ObjectId) -> None:
            now_path = os.path.join(CHECKIN_DIR, timestamp.strftime("%Y-%m-%d"), str(attendance_id))
            if snapshot["frame"] is not None:
                self._save_queue.put((snapshot["frame"], os.path.join(now_path, "frame.jpg")))
            for idx, (_, __, frm, emb) in enumerate(snapshot["frames"]):
                self._save_queue.put((frm, os.path.join(now_path, "frames", f"frame_{idx}.jpg")))

            video_writer = snapshot.get("video_writer")
            if video_writer is not None:
                video_writer.release()
                old = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                new = os.path.join(now_path, "video.mp4")
                os.makedirs(os.path.dirname(new), exist_ok=True)
                if os.path.exists(old):
                    os.rename(old, new)

        if snapshot.get("name") == "Unknown":
            if snapshot.get("frame") is None:
                print("No frame available for unknown tracker")
                return
            if len(snapshot["frames"]) < 10:
                print("Not enough frames for unknown tracker")
                return

        print("Process unknown tracker")
        att_id = ObjectId()
        save_frames(att_id)

        record = Attendance(
            id=att_id,
            time=timestamp,
            student_name="Unknown",
            score=snapshot.get("score", 0),
        )

        if snapshot.get("name") != "Unknown":
            find_att = self.attendance_repo.find({"_id": ObjectId(snapshot["name"])})
            if find_att:
                att = find_att[0]
                if att.student_id:
                    students = self.student_repo.find({"_id": att.student_id})
                    if students:
                        student = students[0]
                        record.student_id = student.id
                        record.student_name = student.name

                        already = self.attendance_repo.find({
                            "student_id": student.id,
                            "time": {"$gte": start_of_today_local()},
                        })
                        if already:
                            return

                        cls = next((c for c in self.list_classrooms if c["_id"] == student.class_id), None)
                        record.student_classroom = cls["name"] if cls else ""

        self.attendance_repo.insert(record)
        self._build_unknown_face(snapshot, record)
        self.on_attendance(record)

    def _build_unknown_face(self, tracker_snapshot: dict, attendance: Attendance) -> None:
        """Placeholder for building unknown-face embedding file."""
        return  # currently disabled upstream

    # ------------------------------------------------------------------
    # Attendance registration (legacy path, kept for compatibility)
    # ------------------------------------------------------------------

    def register_attendance(
        self, student: Student, score: float, frame: np.ndarray, bbox
    ) -> Optional[ObjectId]:
        now = time.time()
        last = self._last_seen.get(student.id, 0)
        if now - last < 10:
            return None

        self._last_seen[student.id] = now
        timestamp = now_local()

        face_crop = crop_face_padded(frame, bbox[0], bbox[1], bbox[2], bbox[3])

        cls = next((c for c in self.list_classrooms if c["_id"] == student.class_id), None)
        record = Attendance(
            student_id=student.id,
            student_name=student.name,
            student_classroom=cls["name"] if cls else "",
            time=timestamp,
            score=score,
        )
        record.id = self.attendance_repo.insert(record)
        self.on_attendance(record)
        return record.id

    # ------------------------------------------------------------------
    # Render helpers (pure cv2, no UI framework dependency)
    # ------------------------------------------------------------------

    def get_render_data(self) -> Tuple[Optional[np.ndarray], list]:
        """Return ``(frame_bgr, rects)`` from the latest detection pass."""
        rects, last_frame = self._latest_faces if self._latest_faces else ([], None)
        if last_frame is None:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy(), []
                return None, []
        return last_frame.copy(), rects

    def build_annotated_frame(self) -> Optional[np.ndarray]:
        """Return a fully annotated + resized BGR frame ready for display."""
        frame, rects = self.get_render_data()
        if frame is None:
            return None

        frame_w, frame_h = frame.shape[1], frame.shape[0]
        frame = annotate_frame(frame, rects)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        stats = [
            (f"FPS: {self._frame_fps:.2f}", 30),
            (f"Size: {frame_w}x{frame_h}", 60),
            (f"Detect speed: {self.current_detect_time * 1000:.1f} ms", 90),
            (f"Faces: {self.current_faces}", 120),
            (f"Valid Faces: {self.current_faces_valid}", 150),
            (f"Frame: {self.frame_count}", 180),
            (f"Detect Frame: {self.detect_frame_count}", 210),
        ]
        frame = annotate_frame(frame, [], stats)

        self.streaming.enqueue(frame.copy())
        return frame

    # ------------------------------------------------------------------
    # Face-crop / avatar helpers (pure cv2)
    # ------------------------------------------------------------------

    @staticmethod
    def load_face_crop_for_record(record: Attendance) -> Optional[np.ndarray]:
        """Load the saved face-crop image for an attendance record."""
        path = get_attendance_frame_path(CHECKIN_DIR, record.time, record.id)
        if path and os.path.exists(path):
            return cv2.imread(os.path.normpath(path))
        return None

    @staticmethod
    def load_student_avatar_bgr(student_id: Optional[ObjectId]) -> Optional[np.ndarray]:
        """Load student avatar as BGR numpy array."""
        path = get_student_avatar_path(AVATAR_DIR, student_id)
        if path and os.path.exists(path):
            return cv2.imread(path)
        return None

    # ------------------------------------------------------------------
    # Save worker
    # ------------------------------------------------------------------

    def _save_worker(self) -> None:
        while True:
            frame, path = self._save_queue.get()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame)
            self._save_queue.task_done()

    # ------------------------------------------------------------------
    # Check / debug
    # ------------------------------------------------------------------

    def dump_tracks(self) -> None:
        for tid, info in self._trackid_to_name.items():
            print(
                f"Track ID: {tid}, "
                f"total frames: {len(info['frames'])}, "
                f"name: {info.get('name', '--')}, "
                f"area: {info.get('area', 0)}, "
                f"score: {info.get('score', 0)}, "
                f"student: {info.get('student', '--')}"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.stop_capture()
        self._release_capture()
