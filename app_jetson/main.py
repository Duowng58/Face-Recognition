"""
Attendance App – Jetson / Headless version (no PySide6).

Runs the same capture → detect → recognize → attendance pipeline
as the PySide6 app but displays output in a cv2.imshow window.

Usage:
    python app_jetson/main.py                     # webcam
    python app_jetson/main.py --source rtsp://...  # RTSP stream
    python app_jetson/main.py --no-stream          # disable RTMP
"""

from __future__ import annotations

import argparse
import os
import sys
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from bson import ObjectId

# ── make project root importable ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app_jetson.config import (
    ANNOY_INDEX_PATH,
    AVATAR_DIR,
    CHECKIN_DIR,
    DEFAULT_RTMP_URL,
    DEFAULT_RTSP_URL,
    EMBEDDING_DIM,
    FACE_DATA_DIR,
    FONT_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAPPING_PATH,
    MIN_BBOX_AREA,
    SIM_THRESHOLD,
    TREE,
    VIDEO_FPS,
)
from app.services.recognition import RecognitionService
from app.services.streaming import StreamingService
from app.utils.cv2_helper import check_blur_laplacian, cv2_putText_utf8
from app.utils.mongodb_access import (
    Attendance,
    AttendanceRepository,
    MongoClientSingleton,
    Student,
    StudentRepository,
    now_local,
    start_of_today_local,
    to_local,
)
from app_jetson.face_tracker import Tracker

WINDOW_NAME = "Attendance – Jetson"


class AttendanceApp:
    """Headless attendance application using cv2.imshow for display."""

    def __init__(self, source: int | str, enable_stream: bool = True) -> None:
        self._source = source
        self._enable_stream = enable_stream

        # ── state ─────────────────────────────
        self._running = False
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_faces: list = []
        self._last_seen: dict[str, float] = {}
        self._trackid_to_name: dict = {}
        self._list_classrooms: list = []

        # ── save queue (background writer) ────
        self._save_queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._save_worker, daemon=True).start()

        # ── services ──────────────────────────
        self._recognition = RecognitionService(
            face_data_dir=FACE_DATA_DIR,
            annoy_index_path=ANNOY_INDEX_PATH,
            mapping_path=MAPPING_PATH,
            embedding_dim=EMBEDDING_DIM,
            tree=TREE,
            sim_threshold=SIM_THRESHOLD,
        )

        self._streaming = StreamingService(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            rtmp_url=DEFAULT_RTMP_URL,
        )
        if not enable_stream:
            self._streaming.set_enabled(False)

        # ── tracker (callback, no Qt) ─────────
        self._tracker = Tracker()
        self._tracker.on_disappeared = self._handle_disappeared

        os.makedirs(CHECKIN_DIR, exist_ok=True)

    # ==================================================================
    # Save worker
    # ==================================================================

    def _save_worker(self) -> None:
        while True:
            frame, path = self._save_queue.get()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, frame)
            self._save_queue.task_done()

    # ==================================================================
    # Recognition helpers
    # ==================================================================

    def _load_recognition_assets(self) -> None:
        self._recognition.load_assets(lambda msg: print(f"[INFO] {msg}"))

    def _build_face(self) -> None:
        self._recognition.build_face(on_complete=lambda msg: print(f"[INFO] {msg}"))

    def _recognize(self, embedding: np.ndarray):
        return self._recognition.recognize(embedding)

    # ==================================================================
    # Tracker disappeared handler
    # ==================================================================

    def _handle_disappeared(self, track_id: int) -> None:
        print(f"[TRACK] Object {track_id} disappeared")
        tracker = self._trackid_to_name.get(track_id)
        if tracker is None:
            return

        tracker_snapshot = {
            "frames": list(tracker.get("frames", [])),
            "frame": tracker.get("frame"),
            "name": tracker.get("name"),
            "score": tracker.get("score", 0),
            "attendance_id": tracker.get("attendance_id"),
            "video_writer": tracker.get("video_writers"),
        }
        self._trackid_to_name.pop(track_id, None)

        def _process() -> None:
            timestamp = now_local()

            def save_frames(attendance_id):
                now_path = os.path.join(CHECKIN_DIR, timestamp.strftime("%Y-%m-%d"), str(attendance_id))
                if tracker_snapshot["frame"] is not None:
                    self._save_queue.put((tracker_snapshot["frame"], os.path.join(now_path, "frame.jpg")))
                for idx, (_, __, frame) in enumerate(tracker_snapshot["frames"]):
                    self._save_queue.put((frame, os.path.join(now_path, "frames", f"frame_{idx}.jpg")))

                vw = tracker_snapshot.get("video_writer")
                if vw is not None:
                    vw.release()
                    old = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                    new = os.path.join(now_path, "video.mp4")
                    os.makedirs(os.path.dirname(new), exist_ok=True)
                    if os.path.exists(old):
                        os.rename(old, new)

            if tracker_snapshot.get("name") == "Unknown":
                if tracker_snapshot.get("frame") is None or len(tracker_snapshot["frames"]) < 10:
                    return
                attendance_repo = AttendanceRepository()
                att_id = ObjectId()
                save_frames(att_id)
                self._build_unknown_face(tracker_snapshot, att)

                att = Attendance(
                    id=att_id,
                    time=timestamp,
                    student_name="Unknown",
                    score=tracker_snapshot.get("score", 0),
                )
                attendance_repo.insert(att)
                print(f"[ATTENDANCE] Unknown saved id={att_id}")
            else:
                att_id = tracker_snapshot.get("attendance_id")
                if att_id is not None:
                    save_frames(att_id)
                else:
                    vw = tracker_snapshot.get("video_writer")
                    if vw is not None:
                        vw.release()
                        old = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                        if os.path.exists(old):
                            os.remove(old)

        threading.Thread(target=_process, daemon=True).start()

    def _build_unknown_face(self, tracker: dict, att: Attendance) -> None:
        embeddings = []
        if len(tracker["frames"]) > 10:
            for _, __, face_crop in tracker["frames"]:
                faces = self._recognition.face_app.get(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                if faces:
                    embeddings.append(faces[0].embedding)
            np.save(os.path.join(FACE_DATA_DIR, f"{att.id}.npy"), embeddings)
            print("[🔁] Rebuild Annoy Index")
            self._build_face()

    # ==================================================================
    # Attendance registration
    # ==================================================================

    def _register_attendance(self, student: Student, score: float,
                             frame: np.ndarray, bbox) -> Optional[ObjectId]:
        now = time.time()
        last_time = self._last_seen.get(student.id, 0)
        if now - last_time < 10:
            return None

        self._last_seen[student.id] = now
        timestamp = now_local()

        y1, y2 = max(bbox[1] - int(abs(bbox[1] - bbox[3]) * 0.2), 0), max(bbox[3] + int(abs(bbox[1] - bbox[3]) * 0.2), 0)
        x1, x2 = max(bbox[0] - int(abs(bbox[0] - bbox[2]) * 0.2), 0), max(bbox[2] + int(abs(bbox[0] - bbox[2]) * 0.2), 0)
        face_crop = frame[y1:y2, x1:x2]

        find_class = next((c for c in self._list_classrooms if c["_id"] == student.class_id), None)
        record = Attendance(
            student_id=student.id,
            student_name=student.name,
            student_classroom=find_class["name"] if find_class else "",
            time=timestamp,
            score=score,
        )

        repo = AttendanceRepository()
        record.id = repo.insert(record)
        print(f"[ATTENDANCE] ✔ {student.name} checked in at {to_local(timestamp).strftime('%H:%M:%S')} (score={score:.2f})")
        return record.id

    # ==================================================================
    # Capture loop (runs in its own thread)
    # ==================================================================

    def _capture_loop(self) -> None:
        is_webcam = isinstance(self._source, int)
        while self._running and self._capture is not None:
            try:
                ret, frame = self._capture.read()
            except cv2.error:
                break
            if not ret:
                time.sleep(0.01)
                continue

            if is_webcam:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            with self._frame_lock:
                self._latest_frame = frame.copy()

        self._running = False

    # ==================================================================
    # Detection worker (runs in its own thread)
    # ==================================================================

    def _detect_worker(self) -> None:
        trackid_saved_known: set = set()
        detect_frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while self._running:
            if self._latest_frame is None:
                time.sleep(0.01)
                continue

            detect_frame_count += 1

            with self._frame_lock:
                frame_copy = self._latest_frame.copy()

            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            faces = self._recognition.face_app.get(rgb)

            rects, embeddings = [], []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                    continue
                rects.append([x1, y1, x2, y2])
                embeddings.append(face.embedding)

            tracks = self._tracker.update(rects, classId="face")
            frame = frame_copy.copy()
            raw_frame = frame_copy.copy()
            rects2: list = []

            for track in tracks:
                x1, y1, x2, y2, track_id, _ = track

                matched_embedding = None
                for rect, emb in zip(rects, embeddings):
                    if abs(x1 - rect[0]) < 15 and abs(y1 - rect[1]) < 15:
                        matched_embedding = emb
                        break
                if matched_embedding is None:
                    continue

                if self._trackid_to_name.get(track_id) is None:
                    h, w = frame.shape[:2]
                    self._trackid_to_name[track_id] = {
                        "name": "Unknown",
                        "score": 0.0,
                        "student": None,
                        "frames": [],
                        "frame": None,
                        "attendance_id": None,
                        "video_writers": cv2.VideoWriter(
                            os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4"),
                            fourcc, VIDEO_FPS, (w, h),
                        ),
                    }
                trk = self._trackid_to_name[track_id]

                name, score = self._recognize(matched_embedding)

                if trk["name"] == "Unknown":
                    trk["name"] = name
                else:
                    if name == "Unknown":
                        name = trk["name"]
                    elif name != trk["name"]:
                        continue

                trk["score"] = score
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"ID:{track_id} {name} {score:.2f}" if name != "Unknown" else f"ID:{track_id} Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                frame_tracker = frame_copy.copy()
                cv2.rectangle(frame_tracker, (x1, y1), (x2, y2), color, 2)
                trk["video_writers"].write(frame_tracker)

                if name != "Unknown":
                    if track_id not in trackid_saved_known:
                        trackid_saved_known.add(track_id)
                        student_repo = StudentRepository()
                        found = student_repo.find({"_id": ObjectId(name)})
                        if found:
                            student = found[0]
                            trk["student"] = student
                            label = f"ID:{track_id} {student.name} {score:.2f}"

                            att_repo = AttendanceRepository()
                            existing = att_repo.find({
                                "student_id": student.id,
                                "time": {"$gte": start_of_today_local()},
                            })
                            if not existing:
                                trk["attendance_id"] = self._register_attendance(
                                    student, score, raw_frame, (x1, y1, x2, y2),
                                )
                    else:
                        student = trk.get("student")
                        if student is not None:
                            label = f"ID:{track_id} {student.name} {score:.2f}"

                # Draw label
                if os.path.exists(FONT_PATH):
                    frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Collect face crops
                face_crop_avatar = frame_copy[
                    max(y1 - int(abs(y1 - y2) * 0.2), 0):max(y2 + int(abs(y1 - y2) * 0.2), 0),
                    max(x1 - int(abs(x1 - x2) * 0.2), 0):max(x2 + int(abs(x1 - x2) * 0.2), 0),
                ]
                face_crop_build = frame_copy[y1:y2, x1:x2]
                if face_crop_build is None or face_crop_build.size == 0:
                    continue

                is_blur, variance = check_blur_laplacian(face_crop_build)
                if not is_blur:
                    if not any(detect_frame_count is c for _, c, _ in trk["frames"]):
                        trk["frames"].append((variance, detect_frame_count, face_crop_build.copy()))
                    if score > trk.get("score", 0):
                        trk["score"] = score
                        trk["frame"] = face_crop_avatar.copy()
                    elif variance > trk.get("variance", 0):
                        trk["frame"] = face_crop_avatar.copy()
                        trk["variance"] = variance

                trk["frames"] = sorted(trk["frames"], key=lambda x: x[0], reverse=True)[:15]
                rects2.append((track_id, color, label, [x1, y1, x2, y2]))

            self._latest_faces = rects2
            time.sleep(0.1)

    # ==================================================================
    # Main render loop (runs on main thread)
    # ==================================================================

    def _render_loop(self) -> None:
        """Display frames in cv2 window. Press 'q' or ESC to quit."""
        while self._running:
            with self._frame_lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # Draw face rects + labels on frame
            for track_id, color, label, (x1, y1, x2, y2) in self._latest_faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if os.path.exists(FONT_PATH):
                    frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow(WINDOW_NAME, frame)
            self._streaming.enqueue(frame.copy())

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                print("[INFO] Quit requested")
                self._running = False
                break

    # ==================================================================
    # Start / Stop
    # ==================================================================

    def start(self) -> None:
        print(f"[INFO] Source: {self._source}")
        print(f"[INFO] Streaming: {'ON' if self._enable_stream else 'OFF'}")

        # Load recognition model
        self._load_recognition_assets()

        # Wait for model to be ready (blocking)
        print("[INFO] Waiting for model to load …")
        while self._recognition.face_app is None:
            time.sleep(0.5)
        print("[INFO] Model ready")

        # Load classrooms
        try:
            client = MongoClientSingleton.get_client()
            self._list_classrooms = list(client.db["classrooms"].find())
        except Exception as e:
            print(f"[WARN] Could not load classrooms: {e}")
            self._list_classrooms = []

        # Open video source
        if isinstance(self._source, int):
            backend = cv2.CAP_ANY
        else:
            backend = cv2.CAP_FFMPEG

        self._capture = cv2.VideoCapture(self._source, backend)
        if not isinstance(self._source, int):
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            print("[ERROR] Cannot open video source!")
            return

        fps = self._capture.get(cv2.CAP_PROP_FPS) or 25
        print(f"[INFO] Video opened – {FRAME_WIDTH}x{FRAME_HEIGHT} @ {fps:.0f} FPS")

        self._running = True

        # Start streaming
        self._streaming.start(fps, lambda: self._running)

        # Start background threads
        cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        det_thread = threading.Thread(target=self._detect_worker, daemon=True)
        cap_thread.start()
        det_thread.start()

        # Render on main thread (blocks until quit)
        try:
            self._render_loop()
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False

        self._streaming.stop()

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        try:
            self._tracker.release()
        except Exception:
            pass

        cv2.destroyAllWindows()
        print("[INFO] Stopped")


# ==================================================================
# CLI entry point
# ==================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Attendance App – Jetson (cv2)")
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Video source: integer for webcam index, or RTSP URL. "
             f"Default: 0 (webcam). RTSP default: {DEFAULT_RTSP_URL}",
    )
    parser.add_argument(
        "--rtsp", action="store_true",
        help=f"Use default RTSP URL: {DEFAULT_RTSP_URL}",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable RTMP live streaming",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
    elif args.rtsp:
        source = DEFAULT_RTSP_URL
    else:
        source = 0

    app = AttendanceApp(source=source, enable_stream=not args.no_stream)
    app.start()


if __name__ == "__main__":
    main()
