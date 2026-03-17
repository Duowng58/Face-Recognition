from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional, Tuple

from bson import ObjectId
import cv2
from matplotlib.pylab import f
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.config import (
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
    VIDEO,
    VIDEO_FOURCC,
    VIDEO_FPS,
)
from app.services.recognition import RecognitionService
from app.services.streaming import StreamingService
from app.ui.dialogs import UpdateStudentDialog
from app.ui.styles import STYLE_SHEET
from app.utils.cv2_helper import check_blur_laplacian, cv2_putText_utf8
from app.utils.image_utils import (
    get_attendance_frame_path,
    get_student_avatar_path,
)
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
from app.utils.qt_invoker import qt_invoke
from app.services.face_tracker import Tracker
# from norfair import Detection, Tracker, draw_tracked_objects


class AttendanceWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hệ thống điểm danh khuôn mặt")
        self.setMinimumSize(1180, 720)
        # self.showFullScreen()

        self._central = QtWidgets.QWidget()
        self.setCentralWidget(self._central)

        root_layout = QtWidgets.QVBoxLayout(self._central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        title = QtWidgets.QLabel("Hệ thống điểm danh khuôn mặt")
        title.setObjectName("title")
        title.setAlignment(QtCore.Qt.AlignCenter)
        root_layout.addWidget(title)

        root_layout.addLayout(self._build_form_row())

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(14)
        root_layout.addLayout(content_layout, stretch=1)

        content_layout.addWidget(self._build_video_panel(), stretch=3)
        content_layout.addWidget(self._build_info_panel(), stretch=2)

        # -- state --
        self._last_seen: dict[str, float] = {}
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # -- recognition service --
        self._recognition = RecognitionService(
            face_data_dir=FACE_DATA_DIR,
            annoy_index_path=ANNOY_INDEX_PATH,
            mapping_path=MAPPING_PATH,
            embedding_dim=EMBEDDING_DIM,
            tree=TREE,
            sim_threshold=SIM_THRESHOLD,
        )

        # -- streaming service --
        self._streaming = StreamingService(
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            rtmp_url=DEFAULT_RTMP_URL,
        )

        # -- tracker --
        self._tracker = Tracker()
        self._tracker.on_disappeared_signal.connect(self._handle_disappeared)

      

        self._trackid_to_name: dict = {}

        self._save_queue: queue.Queue = queue.Queue()

        self._attendance_selected: Optional[Attendance] = None
        self._attendance_selected_row: Optional[int] = None
        self.list_classrooms: list = []

        os.makedirs(CHECKIN_DIR, exist_ok=True)

        self._load_recognition_assets()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._render_frame)

        def save_worker():
            while True:
                frame, path = self._save_queue.get()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, frame)
                self._save_queue.task_done()

        threading.Thread(target=save_worker, daemon=True).start()

        self.setStyleSheet(STYLE_SHEET)

        self.load_params()

    # ==================================================================
    # Recognition helpers (delegate to service)
    # ==================================================================

    def _load_recognition_assets(self) -> None:
        self.status_label.setText("Thông báo: Đang tải mô hình nhận diện...")
        self._recognition.load_assets(
            lambda msg: qt_invoke(lambda: self.status_label.setText(msg))
        )
        

    def _build_face(self) -> None:
        self._recognition.build_face(
            on_complete=lambda msg: qt_invoke(lambda: self.status_label.setText(msg))
        )

    def _recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        return self._recognition.recognize(embedding)

    # ==================================================================
    # Tracker disappeared handler
    # ==================================================================

    def _handle_disappeared(self, track_id: int) -> None:
        print(f"Object {track_id} disappeared")
        tracker = self._trackid_to_name.get(track_id)
        if tracker is None:
            return
        print(f"Track ID: {track_id}")
        # print(tracker)
        print(
            f"total frames: {len(tracker['frames'])}, "
            f"name: {tracker.get('name', '--')}, "
            f"variance: {tracker.get('variance', 0)}, "
            f"score: {tracker.get('score', 0)}, "
            f"student: {tracker.get('student', '--')}"
        )

        tracker_snapshot = {
            "frames": list(tracker.get("frames", [])),
            "frame": tracker.get("frame"),
            "name": tracker.get("name"),
            "score": tracker.get("score", 0),
            "attendance_id": tracker.get("attendance_id"),
            "video_writer": tracker.get("video_writers"),
        }
        self._trackid_to_name.pop(track_id, None)

        def handle_disappeared() -> None:
            timestamp = now_local()

            def save_frames(attendance_id):
                now_path = os.path.join(CHECKIN_DIR, timestamp.strftime("%Y-%m-%d"), str(attendance_id))
                if tracker_snapshot["frame"] is not None:
                    self._save_queue.put((tracker_snapshot["frame"], os.path.join(now_path, "frame.jpg")))
                for index, (_, __, frame, emb) in enumerate(tracker_snapshot["frames"]):
                    self._save_queue.put((frame, os.path.join(now_path, "frames", f"frame_{index}.jpg")))

                video_writer = tracker_snapshot.get("video_writer")
                if video_writer is not None:
                    video_writer.release()
                    old_path = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                    new_path = os.path.join(CHECKIN_DIR, timestamp.strftime("%Y-%m-%d"), str(attendance_id), "video.mp4")
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)

            if tracker_snapshot.get("name") == "Unknown":
                if tracker_snapshot.get("frame") is None:
                    print("No frame available for unknown tracker")
                    return
                if len(tracker_snapshot["frames"]) < 10:
                    print("Not enough frames for unknown tracker")
                    return
                print("Process unknown tracker")
                attendance_repo = AttendanceRepository()
                attendance_unknown_id = ObjectId()
                save_frames(attendance_unknown_id)

                attendance_unknown = Attendance(
                    id=attendance_unknown_id,
                    time=timestamp,
                    student_name="Unknown",
                    score=tracker_snapshot.get("score", 0),
                )
                attendance_repo.insert(attendance_unknown)
                self.build_unknown_face(tracker_snapshot, attendance_unknown)
                qt_invoke(lambda: self._append_history_row(attendance_unknown))
            else:
                attendance_id = tracker_snapshot.get("attendance_id")
                if attendance_id is not None:
                    print("Điểm danh unknown")
                    save_frames(attendance_id)
                else:
                    video_writer = tracker_snapshot.get("video_writer")
                    if video_writer is not None:
                        print("Release video writer, Không điểm danh")
                        video_writer.release()
                        old_path = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    else:
                        print("Do nothing")

        threading.Thread(target=handle_disappeared, daemon=True).start()

    def build_unknown_face(self, tracker: dict, attendance_unknown: Attendance) -> None:
        embeddings = []
        if len(tracker["frames"]) > 10:
            for variance, frame_count, face_crop, emb in tracker["frames"]:
                embeddings.append(emb)
            print('total embeddings:', len(embeddings))
            if len(embeddings) > 0:
                np.save(os.path.join(FACE_DATA_DIR, f"{attendance_unknown.id}.npy"), embeddings)
                print("[🔁] Rebuild Annoy Index")
            # self._build_face()

    # ==================================================================
    # Data loading
    # ==================================================================

    def load_params(self) -> None:
        self._refresh_classrooms()

        attendanceRepo = AttendanceRepository()
        list_attendance = attendanceRepo.find({
            "time": {"$gte": start_of_today_local()}
        })

        self.clear_selected_row()
        while self.history_table.rowCount() > 0:
            self.history_table.removeRow(0)

        for attendance in list_attendance:
            self._append_history_row(attendance)

    def _refresh_classrooms(self) -> None:
        client = MongoClientSingleton.get_client()
        self.list_classrooms = list(client.db["classrooms"].find())
        print(self.list_classrooms)

    def clear_selected_row(self) -> None:
        self._attendance_selected = None
        self.id_value.setText('--')
        self.name_value.setText('--')
        self.class_value.setText('--')
        self.time_value.setText('--')

        self.status_label.setText('-')
        self.avatar.clear()
        self._update_student_image(None)

    # ==================================================================
    # UI builders
    # ==================================================================

    def _build_form_row(self) -> QtWidgets.QLayout:
        row = QtWidgets.QGridLayout()
        row.setHorizontalSpacing(12)
        row.setVerticalSpacing(6)

        source_label = QtWidgets.QLabel("Nguồn video")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP", "Video File"])

        rtsp_label = QtWidgets.QLabel("RTSP URL")
        self.rtsp_edit = QtWidgets.QLineEdit(DEFAULT_RTSP_URL)

        self.stream_toggle = QtWidgets.QCheckBox("Livestream")
        self.stream_toggle.setChecked(True)
        self.stream_toggle.toggled.connect(self._toggle_streaming)

        row.addWidget(source_label, 0, 0)
        row.addWidget(self.source_combo, 0, 1)
        row.addWidget(rtsp_label, 0, 2)
        row.addWidget(self.rtsp_edit, 0, 3, 1, 3)
        row.addWidget(self.stream_toggle, 0, 6)

        row.setColumnStretch(1, 1)
        row.setColumnStretch(3, 1)
        row.setColumnStretch(5, 1)
        row.setColumnStretch(6, 0)
        return row

    def _build_video_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QGroupBox()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setSpacing(10)

        self.video_frame = QtWidgets.QLabel()
        self.video_frame.setObjectName("videoFrame")
        self.video_frame.setAlignment(QtCore.Qt.AlignCenter)
        # self.video_frame.setScaledContents(True)
        panel_layout.addWidget(self.video_frame, stretch=1)

        status_box = QtWidgets.QFrame()
        status_box.setObjectName("statusBox")
        status_layout = QtWidgets.QHBoxLayout(status_box)
        status_layout.setContentsMargins(12, 8, 12, 8)
        self.status_label = QtWidgets.QLabel("Thông báo: Chưa có dữ liệu điểm danh.")
        status_layout.addWidget(self.status_label)
        panel_layout.addWidget(status_box)

        button_row = QtWidgets.QHBoxLayout()

        self.open_btn = QtWidgets.QPushButton("Bắt đầu")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.clicked.connect(self._toggle_capture)

        self.build_face_btn = QtWidgets.QPushButton("Nhập dữ liệu học sinh")
        self.build_face_btn.setObjectName("secondaryButton")
        self.build_face_btn.clicked.connect(self._import_data)

        self.history_btn = QtWidgets.QPushButton("Check")
        self.history_btn.setObjectName("ghostButton")
        self.history_btn.clicked.connect(self._check_things)
        button_row.addWidget(self.open_btn)
        button_row.addWidget(self.build_face_btn)
        button_row.addWidget(self.history_btn)
        panel_layout.addLayout(button_row)

        return panel

    def _import_data(self) -> None:
        print("Importing data...")
        self.load_params()
        self._recognition.build_face(
            on_complete=lambda msg: qt_invoke(lambda: self.status_label.setText(msg))
        )

    def _check_things(self) -> None:
        print("Checking things...")
        for track_id, tracker in self._trackid_to_name.items():
            print(f"Track ID: {track_id}")
            print(
                f'total frames: {len(tracker["frames"])}, '
                f'name: {tracker.get("name", "--")}, '
                f'area: {tracker.get("area", 0)}, '
                f'score: {tracker.get("score", 0)}, '
                f'student: {tracker.get("student", "--")}'
            )

    def _build_info_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QGroupBox()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)

        self.success_label = QtWidgets.QLabel("Thông tin điểm danh")
        self.success_label.setObjectName("successTitle")
        layout.addWidget(self.success_label)

        btn_layout = QtWidgets.QHBoxLayout()
        avatar_column = QtWidgets.QHBoxLayout()
        avatar_column.setSpacing(6)

        self.avatar = QtWidgets.QLabel()
        self.avatar.setFixedSize(120, 120)
        self.avatar.setObjectName("avatar")
        self.avatar.setAlignment(QtCore.Qt.AlignCenter)
        avatar_column.addWidget(self.avatar)

        self.student_image = QtWidgets.QLabel()
        self.student_image.setFixedSize(120, 120)
        self.student_image.setObjectName("avatar")
        self.student_image.setAlignment(QtCore.Qt.AlignCenter)
        avatar_column.addWidget(self.student_image)
        avatar_column.addStretch(1)
        btn_layout.addLayout(avatar_column, stretch=1)

        action_row = QtWidgets.QVBoxLayout()
        self.update_student_btn = QtWidgets.QPushButton("Cập nhật")
        self.update_student_btn.setObjectName("ghostButton")
        self.update_student_btn.clicked.connect(self._open_update_student_dialog)
        action_row.addWidget(self.update_student_btn, alignment=QtCore.Qt.AlignTop)
        btn_layout.addLayout(action_row)

        layout.addLayout(btn_layout)

        info_form = QtWidgets.QFormLayout()
        info_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        info_form.setFormAlignment(QtCore.Qt.AlignTop)

        self.id_value = QtWidgets.QLabel("--")
        self.name_value = QtWidgets.QLabel("--")
        self.class_value = QtWidgets.QLabel("--")
        self.time_value = QtWidgets.QLabel("--")
        info_form.addRow("ID Học sinh:", self.id_value)
        info_form.addRow("Tên Học sinh:", self.name_value)
        info_form.addRow("Lớp:", self.class_value)
        info_form.addRow("Thời gian:", self.time_value)
        layout.addLayout(info_form)

        divider = QtWidgets.QWidget()
        divider.setFixedHeight(1)
        divider.setObjectName("hLine")
        layout.addWidget(divider)

        history_label = QtWidgets.QLabel("Lịch sử điểm danh")
        history_label.setObjectName("historyTitle")
        layout.addWidget(history_label)

        self.history_table = QtWidgets.QTableWidget(0, 4)
        self.history_table.cellClicked.connect(self._on_history_cell_clicked)
        self.history_table.setHorizontalHeaderLabels([
            "Lớp",
            "Tên học sinh",
            "Thời gian",
            "Score",
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        self.history_table.setMinimumHeight(180)
        layout.addWidget(self.history_table, stretch=1)

        return panel

    # ==================================================================
    # History table
    # ==================================================================

    def _on_history_cell_clicked(self, row: int, column: int) -> None:
        self._attendance_selected = self.history_table.item(row, 0).data(QtCore.Qt.UserRole)
        self._attendance_selected_row = row
        if self._attendance_selected is not None:
            self._update_attendance_panel(self._attendance_selected)

    def _append_history_row(self, record: Attendance) -> None:
        row = 0
        self.history_table.insertRow(row)
        student_classroom_item = QtWidgets.QTableWidgetItem(record.student_classroom)
        student_classroom_item.setData(QtCore.Qt.UserRole, record)
        self.history_table.setItem(row, 0, student_classroom_item)
        self.history_table.setItem(row, 1, QtWidgets.QTableWidgetItem(record.student_name))
        self.history_table.setItem(
            row,
            2,
            QtWidgets.QTableWidgetItem(to_local(record.time).strftime("%H:%M:%S")),
        )
        self.history_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{record.score:.2f}"))
        self.history_table.scrollToTop()

    # ==================================================================
    # Update student dialog
    # ==================================================================

    def _open_update_student_dialog(self) -> None:
        if self._attendance_selected is None:
            QtWidgets.QMessageBox.information(self, "Thiếu dữ liệu", "Chưa chọn học sinh để cập nhật.")
            return

        student_repo = StudentRepository()
        attendance_repo = AttendanceRepository()

        student = None
        student_id = self._attendance_selected.student_id
        if student_id is not None:
            try:
                if not isinstance(student_id, ObjectId):
                    student_id = ObjectId(str(student_id))
                student = student_repo.get(student_id)
            except Exception:
                student_id = None

        result = UpdateStudentDialog.show(
            parent=self,
            attendance=self._attendance_selected,
            student=student,
            student_id=student_id,
            classrooms=self.list_classrooms,
            avatar_dir=AVATAR_DIR,
            checkin_dir=CHECKIN_DIR,
        )
        if result is None:
            return

        # A new classroom may have been created inside the dialog
        self._refresh_classrooms()

        if result.selected_student_id is not None:
            student_id = result.selected_student_id

        if student_id is None:
            student = Student(
                id=self._attendance_selected.id,
                name=result.new_name,
                class_id=result.new_class_id,
            )
            student_id = student_repo.insert(student)
            os.makedirs(AVATAR_DIR, exist_ok=True)
            if result.attendance_avatar_pixmap is not None:
                result.attendance_avatar_pixmap.save(
                    os.path.join(AVATAR_DIR, f"{student_id}.jpg")
                )
        else:
            student_repo.update(student_id, {
                "name": result.new_name,
                "class_id": result.new_class_id,
            })

        attendance_repo.update(self._attendance_selected.id, {
            "student_id": student_id,
            "student_name": result.new_name,
            "student_classroom": result.class_name,
        })

        self.name_value.setText(result.new_name)
        self.class_value.setText(result.class_name)

        if self._attendance_selected is not None:
            self._attendance_selected.student_name = result.new_name
            self._attendance_selected.student_classroom = result.class_name
            self._attendance_selected.student_id = student_id

        self._update_student_image(student_id)

        if self._attendance_selected_row is not None:
            self.history_table.item(self._attendance_selected_row, 0).setText(result.class_name)
            self.history_table.item(self._attendance_selected_row, 1).setText(result.new_name)

    # ==================================================================
    # Capture / Video
    # ==================================================================

    def _toggle_capture(self) -> None:
        if self._running:
            qt_invoke(self._stop_capture)
        else:
            qt_invoke(self._start_capture)

    def _start_capture(self) -> None:
        if self._running:
            return

        source: int | str = 0
        if self.source_combo.currentText() == "RTSP":
            source = self.rtsp_edit.text().strip()
            if not source:
                QtWidgets.QMessageBox.warning(self, "Thiếu RTSP", "Vui lòng nhập RTSP URL.")
                return
        elif self.source_combo.currentText() == "Video File":
            source = VIDEO
        if source == 0:
            try:
                temp_cap = cv2.VideoCapture(0)
                if temp_cap.isOpened():
                    temp_cap.release()
            except Exception as e:
                print(f"Error opening camera {0}: {e}")
        backend = cv2.CAP_ANY if source == 0 else cv2.CAP_FFMPEG
        self._capture = cv2.VideoCapture(source, backend)
        if source != 0 :
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Không mở được nguồn video.")
            return
        print(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),self._capture.get(cv2.CAP_PROP_FPS))
        self._Frame_FPS = self._capture.get(cv2.CAP_PROP_FPS)
        self._running = True
        self.open_btn.setText("Kết thúc")

        self._latest_faces: list = []

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        self._detect_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._detect_thread.start()

        # Start streaming
        fps = self._Frame_FPS or 25
        self._streaming.start(fps, lambda: self._running)
        self._timer.start()

    def _toggle_streaming(self, enabled: bool) -> None:
        fps = getattr(self, "_Frame_FPS", None) or 25
        self._streaming.toggle(enabled, fps, lambda: self._running)

    def _stop_capture(self) -> None:
        if not self._running and self._capture is None:
            return

        self._running = False
        self.open_btn.setText("Bắt đầu")
        self._timer.stop()

        for thread in (self._capture_thread, self._detect_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1)

        self._streaming.stop()
        self._release_capture()

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
            if hasattr(self._tracker, "release"):
                self._tracker.release()
        except Exception:
            pass

        self.video_frame.clear()

    def _capture_loop(self) -> None:
        while self._running and self._capture is not None:
            try:
                ret, frame = self._capture.read()
            except cv2.error:
                self._stop_capture()
                break
            if not ret:
                time.sleep(0.01)
                continue

            if self.source_combo.currentText() == "Webcam":
                frame = cv2.flip(frame, 1)
                
            # frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            with self._frame_lock:
                self._latest_frame = frame
            time.sleep(1 / (self._Frame_FPS or 25))
            # time.sleep(1 / 25)

        self._running = False

    # ==================================================================
    # Detection worker
    # ==================================================================

    def _detect_worker(self) -> None:
        trackid_saved_known: set = set()
        detect_frame_count = 0

        while self._running:
            if self._latest_frame is None:
                continue
            
            # Tính thời gian detect
            start_time = time.time()
            detect_frame_count += 1
            print(f"Processing frame {detect_frame_count}")

            with self._frame_lock:
                frame_copy = self._latest_frame.copy()
            rgb = frame_copy
            try:
                start_time = time.time()
                faces = self._recognition.face_app.get(rgb)
                end_time = time.time()
                print(f"Face detection time: {end_time - start_time:.2f} seconds for {len(faces)} faces")
            except Exception as e:
                print(f"Error during face detection: {e}")
                continue
            rects: list = []
            embeddings: list = []
            confidences = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue
                
                mask_label, mask_conf = self._recognition.check_mask(frame_copy[y1:y2, x1:x2])
                if mask_label == "Mask":
                    continue

                rects.append([x1, y1, x2, y2])
                embeddings.append(face.normed_embedding)
                confidences.append(face.det_score)
                # print(f"Mask detection: {mask_label} ({mask_conf:.2f})",face.crop)
            print(f"Processed {len(embeddings)} faces valid")
            rects2: list = []

            
            
            tracks = self._tracker.update(rects, classId="face")
            # frame = frame_copy.copy()
            raw_frame = frame_copy.copy()
            # start_time = time.time()
            
            for track in tracks:
                x1, y1, x2, y2, track_id, _ = track

                matched_embedding = None

                for rect, emb, conf in zip(rects, embeddings, confidences):
                    rx1, ry1, rx2, ry2 = rect

                    if abs(x1 - rx1) < 15 and abs(y1 - ry1) < 15:
                        matched_embedding = (emb, rect, conf)
                        break

                if matched_embedding is None:
                    continue
                if self._trackid_to_name.get(track_id) is None:
                    # h, w = frame.shape[:2]
                    self._trackid_to_name[track_id] = {
                        "name": "Unknown",
                        "score": 0.0,
                        "student": None,
                        "frames": [],
                        "frame": None,
                        "attendance_id": None,
                        # "video_writers": cv2.VideoWriter(
                        #     os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4"),
                        #     VIDEO_FOURCC, VIDEO_FPS, (w, h),
                        # ),
                    }
                tracker = self._trackid_to_name.get(track_id)
                # print('start recognize')
                name, score = self._recognize(matched_embedding[0])
                # print('end recognize')
                
                if tracker["name"] == "Unknown":
                    # Kiểm tra tracker hiện tại mà đang Unknow thì lấy theo name detect được
                    tracker["name"] = name
                    tracker["score"] = score
                else:
                    if name == "Unknown":
                        # nếu tracker hiện tại mà đã được detect rồi mà lần sau lại detect lại không ra thì lấy lại name cũ
                        name = tracker["name"]
                        score = tracker["score"]
                    elif name != tracker["name"]:
                        # nếu tracker hiện tại mà đã được detect rồi mà lần sau lại detect lại khác tên (ra người khác) thì kiểm tra score
                        if score > tracker["score"]:
                            # nếu score mới lớn hơn thì cập nhật tên và score và làm mới frames
                            tracker["name"] = name
                            tracker["score"] = score
                            tracker["frames"] = []
                        else:
                            # nếu score mới nhỏ hơn hoặc bằng thì giữ nguyên tên và score cũ
                            name = tracker["name"]
                            score = tracker["score"]

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = (
                    f"ID:{track_id} - {score:.2f} - {matched_embedding[2]:.2f}"
                    if name != "Unknown"
                    else f"ID:{track_id} Unknown"
                )
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # frame_tracker = frame_copy.copy()
                # cv2.rectangle(frame_tracker, (x1, y1), (x2, y2), color, 2)

                # tracker["video_writers"].write(frame_tracker)

                if name != "Unknown":
                    if track_id not in trackid_saved_known:
                        trackid_saved_known.add(track_id)

                        studentRep = StudentRepository()
                        findStudent = studentRep.find({"_id": ObjectId(name)})
                        if len(findStudent) > 0:
                            student = findStudent[0]
                            tracker["student"] = student
                            label = f"ID:{track_id} {student.name} {score:.2f}"

                            attendanceRepo = AttendanceRepository()
                            findStudent = attendanceRepo.find({
                                "student_id": student.id,
                                "time": {"$gte": start_of_today_local()}
                            })
                            if len(findStudent) == 0:
                                tracker["attendance_id"] = self._register_attendance(
                                    student, score, raw_frame, (x1, y1, x2, y2),
                                )
                    else:
                        student = tracker.get("student")
                        if student is not None:
                            label = f"ID:{track_id} {student.name} {score:.2f}"

                # if os.path.exists(FONT_PATH):
                #     frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
                # else:
                #     cv2.putText(frame, label, (x1, y1 - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if frame_copy is not None:
                    face_crop_avatar = frame_copy[
                        max(y1 - int((abs(y1 - y2)) * 0.2), 0):max(y2 + int((abs(y1 - y2)) * 0.2), 0),
                        max(x1 - int((abs(x1 - x2)) * 0.2), 0):max(x2 + int((abs(x1 - x2)) * 0.2), 0),
                    ]
                    
                    # is_blur, variance = check_blur_laplacian(face_crop_build)
                    # print(f"Track ID: {track_id}, Blur: {is_blur}, Variance: {variance}, Area: {face_crop_build.shape[0] * face_crop_build.shape[1]}")   
                    is_blur = False
                    detect_score = matched_embedding[2]
                    if not is_blur:
                        if not any(detect_frame_count is c for _, c, f, e in tracker["frames"]):
                            # Kiểm tra nếu frame này chưa được thêm vào tracker["frames"]
                            tracker["frames"].append((detect_score, detect_frame_count, face_crop_avatar, matched_embedding[0]))
                            
                        if detect_score > tracker.get("variance", 0):
                            # Nếu detect_score mới lớn hơn thì cập nhật frame (Ảnh đại diện) và variance
                            tracker["frame"] = face_crop_avatar.copy()
                            tracker["variance"] = detect_score

                tracker["frames"] = sorted(tracker["frames"], key=lambda x: x[0], reverse=True)[:15]
                rects2.append((track_id, color, label, [x1, y1, x2, y2], matched_embedding[2]))
            # end_time = time.time()
            # print(f'End processing tracks, time taken: {end_time - start_time:.2f} seconds')
            self._latest_faces = rects2, raw_frame
            
            end_time = time.time()
            print(f"Face detection time: {end_time - start_time:.2f} seconds")
            # time.sleep(0.1)

    # ==================================================================
    # Attendance
    # ==================================================================

    def _register_attendance(self, student: Student, score: float, frame: np.ndarray, bbox) -> Optional[ObjectId]:
        now = time.time()
        last_time = self._last_seen.get(student.id, 0)
        if now - last_time < 10:
            return None

        self._last_seen[student.id] = now
        timestamp = now_local()

        face_crop = frame[
            max(bbox[1] - int((abs(bbox[1] - bbox[3])) * 0.2), 0):max(bbox[3] + int((abs(bbox[1] - bbox[3])) * 0.2), 0),
            max(bbox[0] - int((abs(bbox[0] - bbox[2])) * 0.2), 0):max(bbox[2] + int((abs(bbox[0] - bbox[2])) * 0.2), 0),
        ]

        find_class = next((c for c in self.list_classrooms if c["_id"] == student.class_id), None)
        print(find_class)
        record = Attendance(
            student_id=student.id,
            student_name=student.name,
            student_classroom=find_class["name"] if find_class else "",
            time=timestamp,
            score=score,
        )

        attendanceRep = AttendanceRepository()
        record.id = attendanceRep.insert(record)

        qt_invoke(lambda: self._update_attendance_panel(record, face_crop))
        qt_invoke(lambda: self._append_history_row(record))

        return record.id

    def _update_attendance_panel(self, record: Attendance, face_crop: np.ndarray = None) -> None:
        self.id_value.setText(str(record.student_id))
        self.name_value.setText(record.student_name)
        self.class_value.setText(record.student_classroom)
        self.time_value.setText(to_local(record.time).strftime("%H:%M:%S"))

        status = (
            f"Thông báo: {record.student_name} đã điểm danh lúc {to_local(record.time).strftime('%H:%M:%S')}."
        )
        self.status_label.setText(status)

        if face_crop is None:
            frame_path = get_attendance_frame_path(CHECKIN_DIR, record.time, record.id)
            if frame_path and os.path.exists(frame_path):
                face_crop = cv2.imread(os.path.normpath(frame_path))

        if face_crop is not None and face_crop.size > 0:
            target_size = self.avatar.size()

            if target_size.width() <= 0:
                target_size = QtCore.QSize(120, self.avatar.height())

            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(image)

            scaled = pixmap.scaled(
                target_size,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )

            canvas = QtGui.QPixmap(target_size)
            canvas.fill(QtCore.Qt.transparent)
            painter = QtGui.QPainter(canvas)
            x = (target_size.width() - scaled.width()) // 2
            y = (target_size.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            painter.end()

            self.avatar.setPixmap(scaled)
        else:
            self.avatar.clear()

        self._update_student_image(record.student_id)

    def _update_student_image(self, student_id: Optional[ObjectId]) -> None:
        if student_id is None:
            self.student_image.clear()
            return

        image_path = get_student_avatar_path(AVATAR_DIR, student_id)
        if not image_path or not os.path.exists(image_path):
            self.student_image.clear()
            return

        target_size = self.student_image.size()
        if target_size.width() <= 0:
            target_size = QtCore.QSize(120, self.student_image.height())

        pixmap = QtGui.QPixmap(image_path).scaled(
            target_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.student_image.setPixmap(pixmap)

    # ==================================================================
    # Render
    # ==================================================================

    def _render_frame(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is None:
            return

        rects, last_frame = self._latest_faces if self._latest_faces else ([], None)
        frame = last_frame if last_frame is not None else frame
        for track_id, color, label, (x1, y1, x2, y2), detect_sorce in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if os.path.exists(FONT_PATH):
                frame = cv2_putText_utf8(frame, label, (x1, y1 - 40), FONT_PATH, 30, color)
            else:
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_frame.setPixmap(pixmap.scaled(
            self.video_frame.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        self._streaming.enqueue(frame.copy())

    def _show_history(self) -> None:
        self.history_table.scrollToBottom()

    # ==================================================================
    # Close
    # ==================================================================

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        print("app closing")
        self._stop_capture()
        self._release_capture()
        event.accept()
