import json
import os
import queue
import sys
import threading
import time
from typing import List, Optional, Tuple

from bson import ObjectId
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


from insightface.app import FaceAnalysis
from annoy import AnnoyIndex
import subprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    VIDEO_FOURCC,
    VIDEO_FPS,
)
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
from checkin.face_tracker import Tracker
from app.utils.qt_invoker import init_qt_invoker, qt_invoke


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

        # root_layout.addWidget(self._build_footer())

        self._last_seen: dict[str, float] = {}
        self._capture: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._face_app: Optional[FaceAnalysis] = None
        self._annoy_index: Optional[AnnoyIndex] = None
        self._idx2name: dict[str, str] = {}
        
        self._tracker = Tracker()
        self._tracker.on_disappeared_signal.connect(self._handle_disappeared)
        
        self._trackid_to_name = {}
        
        self._save_queue = queue.Queue()
        self._livestream_queue = queue.Queue(maxsize=10)
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._live_stream_thread: Optional[threading.Thread] = None
        self._streaming_enabled = True
        
        self._attendance_selected: Optional[Attendance] = None
        self._attendance_selected_row: Optional[int] = None
        self.list_classrooms = []

        os.makedirs(CHECKIN_DIR, exist_ok=True)

        self._load_recognition_assets()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._render_frame)
        
        def save_worker():
            while True:
                frame, path = self._save_queue.get()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                # print(f"Saving frame to {path}")
                cv2.imwrite(path, frame)
                self._save_queue.task_done()

        threading.Thread(target=save_worker, daemon=True).start()

        self._apply_styles()
        
        self.load_params()
        
    def _handle_disappeared(self, track_id: int) -> None:
        print(f"Object {track_id} disappeared")
        tracker = self._trackid_to_name.get(track_id)
        if tracker is None:
            return
        print(f"Track ID: {track_id}")
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
                for index, (_, __, frame) in enumerate(tracker_snapshot["frames"]):
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
                    return
                if len(tracker_snapshot["frames"]) < 10:
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
        # Implement the logic to build unknown face
        embeddings = []
        if len(tracker["frames"]) > 10:
            for variance, frame_count, face_crop in tracker["frames"]:
                faces = self._face_app.get(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                if faces:
                    embeddings.append(faces[0].embedding)
                    
            np.save(os.path.join(FACE_DATA_DIR, f"{attendance_unknown.id}.npy"), embeddings)
            print("[🔁] Rebuild Annoy Index")
            self._build_face()
            # load lại face_app
            # self._init_recognition()
        pass

    def load_params(self) -> None:
        self._refresh_classrooms()
        
        attendanceRepo = AttendanceRepository()
        list_attendance = attendanceRepo.find({
            "time": {"$gte": start_of_today_local()}
        })
        
        self.clear_selected_row()
        # clear self.history_table data
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
        
        # Implement the logic to clear the selected row
        pass

    def _build_form_row(self) -> QtWidgets.QLayout:
        row = QtWidgets.QGridLayout()
        row.setHorizontalSpacing(12)
        row.setVerticalSpacing(6)

        source_label = QtWidgets.QLabel("Nguồn video")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP"])

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
        # self.video_frame.setPixmap(build_placeholder_pixmap(QtCore.QSize(720, 420), "Name"))
        self.video_frame.setScaledContents(True)
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
        # Implement your data import logic here

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
        # panel.setFixedWidth(600)
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
    
    def _on_history_cell_clicked(self, row: int, column: int) -> None:
        self._attendance_selected = self.history_table.item(row, 0).data(QtCore.Qt.UserRole)
        self._attendance_selected_row = row
        if self._attendance_selected is not None:
            self._update_attendance_panel(self._attendance_selected)
            # print(self._attendance_selected)

    def _open_update_student_dialog(self) -> None:
        if self._attendance_selected is None:
            QtWidgets.QMessageBox.information(self, "Thiếu dữ liệu", "Chưa chọn học sinh để cập nhật.")
            return

        def get_avatar(path: str) -> QtGui.QPixmap:
            print(path)
            if not path or not os.path.exists(path):
                print("File does not exist")
                return QtGui.QPixmap()
            pixmap = QtGui.QPixmap(path).scaled(180, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            return pixmap
        
        def update_student_avatar(student_id: ObjectId) -> None:
            image_path = get_student_avatar_path(AVATAR_DIR, student_id)
            if image_path and os.path.exists(image_path):
                image = get_avatar(image_path)
                student_avatar.setPixmap(image)
            else:
                student_avatar.clear()
                
        def update_attendance_avatar(attendance_id: ObjectId) -> None:
            image_path = get_attendance_frame_path(CHECKIN_DIR, self._attendance_selected.time, attendance_id)
            if image_path and os.path.exists(image_path):
                image = get_avatar(image_path)
                attendance_avatar.setPixmap(image)
            else:
                attendance_avatar.clear()
                
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

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Cập nhật học sinh")
        dialog.setModal(True)

        form = QtWidgets.QFormLayout(dialog)
        student_avatar_box = QtWidgets.QWidget()
        student_avatar_box_layout = QtWidgets.QVBoxLayout(student_avatar_box)
        student_avatar_box_layout.setContentsMargins(0, 0, 0, 0)
        student_avatar_box_layout.addWidget(QtWidgets.QLabel("Ảnh học sinh"))
        student_avatar = QtWidgets.QLabel()
        student_avatar.setObjectName("avatar")
        student_avatar.setFixedSize(180, 180)
        student_avatar.setAlignment(QtCore.Qt.AlignCenter)
        student_avatar_box_layout.addWidget(student_avatar)
        
        attendance_avatar_box = QtWidgets.QWidget()
        attendance_avatar_box_layout = QtWidgets.QVBoxLayout(attendance_avatar_box)
        attendance_avatar_box_layout.setContentsMargins(0, 0, 0, 0)
        attendance_avatar_box_layout.addWidget(QtWidgets.QLabel("Ảnh điểm danh"))
        attendance_avatar = QtWidgets.QLabel()
        attendance_avatar.setObjectName("avatar")
        attendance_avatar.setFixedSize(180, 180)
        attendance_avatar.setAlignment(QtCore.Qt.AlignCenter)
        attendance_avatar_box_layout.addWidget(attendance_avatar)
        # if student and student.images:
        #     image_path = os.path.normpath(student.images)
        #     if not os.path.isabs(image_path):
        #         image_path = os.path.join(ROOT_DIR, image_path)
        #     if os.path.exists(image_path):
        #         pixmap = QtGui.QPixmap(image_path).scaled(48, 48, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        #         avatar.setPixmap(pixmap)
        images_row = QtWidgets.QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.addWidget(student_avatar_box)
        images_row.addWidget(attendance_avatar_box)
        update_student_avatar(student_id)
        update_attendance_avatar(self._attendance_selected.id)

        id_edit = QtWidgets.QLineEdit(str(student.id) if student else "--")
        id_edit.setStyleSheet("QLineEdit { background:#2a2a2a; color: gray; }")
        id_edit.setReadOnly(True)
        name_edit = QtWidgets.QLineEdit(student.name if student else self._attendance_selected.student_name)
        
        students = student_repo.find()
        selected_student = {"value": None}
        select_student_btn = QtWidgets.QPushButton("Chọn")
        select_student_btn.setObjectName("primaryButton")

        class_combo = QtWidgets.QComboBox()
        class_combo.addItem("Chọn lớp...", None)
        
        for classroom in self.list_classrooms:
            class_combo.addItem(classroom.get("name", ""), classroom.get("_id"))

        class_combo.addItem("+ Thêm lớp mới", "__new__")
        new_class_edit = QtWidgets.QLineEdit()
        new_class_edit.setPlaceholderText("Nhập tên lớp mới")
        new_class_edit.setVisible(False)

        def set_class_selection(class_id) -> None:
            if class_id is None:
                return
            for idx in range(class_combo.count()):
                if str(class_combo.itemData(idx)) == str(class_id):
                    class_combo.setCurrentIndex(idx)
                    break

        if student and student.class_id is not None:
            set_class_selection(student.class_id)

        def handle_student_pick() -> None:
            picked = self._open_student_picker(students)
            if picked is None:
                return
            selected_student["value"] = picked
            name_edit.setText(picked.name)
            set_class_selection(picked.class_id)
            update_student_avatar(picked.id)
            

        select_student_btn.clicked.connect(handle_student_pick)

        def handle_class_change() -> None:
            is_new = class_combo.currentData() == "__new__"
            new_class_edit.setVisible(is_new)

        class_combo.currentIndexChanged.connect(handle_class_change)
        handle_class_change()

        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(id_edit, stretch=1)
        picker_row.addWidget(select_student_btn)
        form.addRow("", images_row)
        form.addRow("ID:", picker_row)
        form.addRow("Tên:", name_edit)
        class_row = QtWidgets.QVBoxLayout()
        class_row.addWidget(class_combo)
        class_row.addWidget(new_class_edit)
        form.addRow("Lớp:     ", class_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        global new_name, selected_student_id, new_class_id, class_name
        new_name = ''
        selected_student_id = None
        new_class_id = None
        class_name = None
        
        def on_accept():
            global new_name, selected_student_id, new_class_id, class_name
            new_name = name_edit.text().strip()
            selected_student_id = selected_student["value"].id if selected_student["value"] is not None else None
            new_class_id = class_combo.currentData()
            if not new_name:
                QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên học sinh.")
                return

            if new_class_id == "__new__":
                new_class_name = new_class_edit.text().strip()
                if not new_class_name:
                    QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên lớp mới.")
                    return
                client = MongoClientSingleton.get_client()
                result = client.db["classrooms"].insert_one({"name": new_class_name})
                new_class_id = result.inserted_id
                self._refresh_classrooms()
                class_name = new_class_name
            elif new_class_id is None:
                QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng chọn lớp.")
                return
            else:
                class_name = class_combo.currentText()
            dialog.accept()
            
        buttons.accepted.connect(on_accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        

        if selected_student_id is not None:
            student_id = selected_student_id

        if student_id is None:
            # Sử dụng thông tin từ self._attendance_selected làm id cho học sinh mới
            student = Student(id=self._attendance_selected.id, name=new_name, class_id=new_class_id)
            student_id = student_repo.insert(student)
            if not os.path.exists(AVATAR_DIR):
                os.makedirs(AVATAR_DIR)
            attendance_avatar.pixmap().save(os.path.join(AVATAR_DIR, f"{student_id}.jpg"))
            # self._save_queue.put
        else:
            student_repo.update(student_id, {"name": new_name, "class_id": new_class_id})

        attendance_repo.update(self._attendance_selected.id, {
            "student_id": student_id,
            "student_name": new_name,
            "student_classroom": class_name
        })

        self.name_value.setText(new_name)
        self.class_value.setText(class_name)

        if self._attendance_selected is not None:
            self._attendance_selected.student_name = new_name
            self._attendance_selected.student_classroom = class_name
            self._attendance_selected.student_id = student_id

        self._update_student_image(student_id)

        if self._attendance_selected_row is not None:
            self.history_table.item(self._attendance_selected_row, 0).setText(class_name)
            self.history_table.item(self._attendance_selected_row, 1).setText(new_name)

    def _open_student_picker(self, students: list[Student]) -> Optional[Student]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Chọn học sinh")
        dialog.setModal(True)

        layout = QtWidgets.QVBoxLayout(dialog)
        table = QtWidgets.QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["ID", "Lớp", "Tên", "Ảnh"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        class_map = {str(c.get("_id")): c.get("name", "") for c in self.list_classrooms}

        for student in students:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(student.id)))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(class_map.get(str(student.class_id), "")))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(student.name))

            image_item = QtWidgets.QTableWidgetItem("--")
            image_path = get_student_avatar_path(AVATAR_DIR, student.id)
            if image_path and os.path.exists(image_path):
                pixmap = QtGui.QPixmap(image_path).scaled(120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                image_item = QtWidgets.QTableWidgetItem()
                image_item.setIcon(QtGui.QIcon(pixmap))
                image_item.setSizeHint(QtCore.QSize(120, 120))
            table.setItem(row, 3, image_item)
            table.item(row, 0).setData(QtCore.Qt.UserRole, student)

        layout.addWidget(table)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        def accept_on_double_click() -> None:
            dialog.accept()

        table.itemDoubleClicked.connect(lambda _: accept_on_double_click())

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None

        current_row = table.currentRow()
        if current_row < 0:
            return None

        student = table.item(current_row, 0).data(QtCore.Qt.UserRole)
        return student


    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-family: "Segoe UI";
                font-size: 11pt;
                color: #e5e7eb;
                background-color: #1f1f1f;
            }
            #title {
                font-size: 18pt;
                font-weight: 700;
                color: #d9e2ef;
            }
            QLineEdit, QComboBox {
                background: #1f1f1f;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e5e7eb;
            }
            QComboBox::drop-down {
                subcontrol-position: center right;
                width: 20px;
                border: 1px solid #CCCCCC;
                border-radius: 2px;
                background: #f0f0f0;
                margin: 1px;
            }
            
            QComboBox::down-arrow {
                width: 0px;
                height: 0px;
                border-left: 6px solid #f0f0f0;
                border-right: 6px solid #f0f0f0;
                border-top: 8px solid #333333;
                margin-top: 2px;
            }
            
            QComboBox::down-arrow:hover {
                border-top-color: #2196F3;
            }
            
            QGroupBox {
                border: 1px solid #2f2f2f;
                border-radius: 12px;
                background: #232323;
                padding: 12px;
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #cbd5f5;
            }
            #videoFrame {
                border-radius: 10px;
                background: #151515;
                min-height: 420px;
            }
            #statusBox {
                background: #1c3a2d;
                border: 1px solid #256c4d;
                border-radius: 6px;
                color: #d1fae5;
            }
            #successTitle {
                font-size: 12pt;
                font-weight: 700;
                color: #34d399;
            }
            #historyTitle {
                font-size: 11pt;
                font-weight: 700;
                color: #93c5fd;
            }
            #hLine {
                background: #2f2f2f;
                height: 1px;
            }
            #avatar {
                border: 1px dashed #3a3a3a;
                border-radius: 6px;
                background: #1a1a1a;
            }
            #primaryButton {
                background: #2563eb;
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
            }
            #primaryButton:hover { background: #1d4ed8; }
            
            #secondaryButton{
                background: #6b7280;
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: 600;
            }
            #secondaryButton:hover { background: #4b5563; }
            #ghostButton {
                background: #2f2f2f;
                color: #e5e7eb;
                padding: 8px 16px;
                border-radius: 8px;
            }
            QLabel {
                background: transparent;
            }
            """
        )

    def _load_recognition_assets(self) -> None:
        self.status_label.setText("Thông báo: Đang tải mô hình nhận diện...")
        threading.Thread(target=self._init_recognition, daemon=True).start()

    def _init_recognition(self) -> None:
        self._face_app = FaceAnalysis(name="buffalo_s", 
                                      providers=["CUDAExecutionProvider","CPUExecutionProvider"],
                                      allowed_modules=["detection", "recognition"])
        self._face_app.prepare(ctx_id=0,det_thresh=0.5, det_size=(320, 320))
        
        if os.path.exists(ANNOY_INDEX_PATH) and os.path.exists(MAPPING_PATH):
            self._annoy_index = AnnoyIndex(EMBEDDING_DIM, "angular")
            self._annoy_index.load(ANNOY_INDEX_PATH)
            with open(MAPPING_PATH, "r", encoding="utf-8") as f:
                self._idx2name = json.load(f)
            total_face = [info for f,info in self._idx2name.items()]
            # count total unique faces
            unique_faces = set(total_face)
            message = f"Thông báo: Đã tải {len(unique_faces)} khuôn mặt đã biết."
        else:
            message = "Thông báo: Chưa có dữ liệu khuôn mặt."
            
        print(message)
        def update_status():
            self.status_label.setText(message)
        qt_invoke(update_status)

    def _build_face(self) -> None:
        # Implement the logic for building face data here
        
        files = [f for f in os.listdir(FACE_DATA_DIR) if f.endswith(".npy")]
        if len(files) == 0:
            return
        ann = AnnoyIndex(EMBEDDING_DIM, "angular")
        idx2name = {}
        idx = 0

        for f in files:
            name = f.replace(".npy", "")
            data = np.load(os.path.join(FACE_DATA_DIR, f))
            for v in data:
                ann.add_item(idx, v)
                idx2name[idx] = name
                idx += 1

        ann.build(TREE)
        if self._annoy_index is not None:
            self._annoy_index.unload()
        ann.save(os.path.join(FACE_DATA_DIR, "face_index.ann"))
        
        with open(os.path.join(FACE_DATA_DIR, "image_paths.json"), "w", encoding="utf-8") as f:
            json.dump(idx2name, f, ensure_ascii=False, indent=2)
        self._load_recognition_assets()

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
        if source == 0:
            try:
                temp_cap = cv2.VideoCapture(0)
                if temp_cap.isOpened():
                    temp_cap.release()
            except Exception as e:
                print(f"Error opening camera {0}: {e}")
        backend = cv2.CAP_ANY if source == 0 else cv2.CAP_FFMPEG
        self._capture = cv2.VideoCapture(source,backend)
        if source != 0:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Không mở được nguồn video.")
            return
        print(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._Frame_FPS = self._capture.get(cv2.CAP_PROP_FPS)
        self._running = True
        self.open_btn.setText("Kết thúc")
        
        # self.latest_frame = None
        self._latest_faces = []
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        self._detect_thread = threading.Thread(target=self.detect_worker, daemon=True)
        self._detect_thread.start()
        
        # Implement the logic for live streaming here
        self._start_streaming()
        self._timer.start()

    def _start_streaming(self) -> None:
        if not self._streaming_enabled or not DEFAULT_RTMP_URL:
            return

        fps = self._Frame_FPS or 25
        command = [
            'ffmpeg',
            '-re',  # Đọc với tốc độ thời gian thực
            '-f', 'rawvideo',  # Định dạng đầu vào là raw video
            '-pix_fmt', 'bgr24',  # Pixel format từ OpenCV
            '-s', f'{FRAME_WIDTH}x{FRAME_HEIGHT}',  # Kích thước frame
            '-r', str(fps),  # Frame rate
            '-i', 'pipe:0',  # Đọc từ stdin
            '-c:v', 'libx264',  # Encode H264
            '-preset', 'ultrafast',
            '-f', 'flv',  # RTMP uses FLV muxer
            DEFAULT_RTMP_URL
        ]
        try:
            self._ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"FFmpeg start failed: {exc}")
            self._ffmpeg_process = None
            self._streaming_enabled = False
            return

        time.sleep(0.2)
        if self._ffmpeg_process.poll() is not None:
            stderr = self._ffmpeg_process.stderr.read().decode("utf-8", errors="ignore") if self._ffmpeg_process.stderr else ""
            print(f"FFmpeg exited early. Live stream disabled.\n{stderr}")
            self._ffmpeg_process = None
            self._streaming_enabled = False
            return

        self._live_stream_thread = threading.Thread(target=self._live_stream_worker, daemon=True)
        self._live_stream_thread.start()

    def _toggle_streaming(self, enabled: bool) -> None:
        self._streaming_enabled = enabled
        if not enabled:
            self._stop_streaming()
            return
        if self._running and self._ffmpeg_process is None:
            self._start_streaming()
        
    def _live_stream_worker(self):
        counter = 0
        print('Live stream worker started.')
        while self._running:
            try:
                # print('Begin get frame for stream')
                frame = self._livestream_queue.get(timeout=0.5)
                # print('Got frame for stream')
            except queue.Empty:
                # print('Live stream queue is empty.')
                continue

            try:
                counter += 1
                if self._ffmpeg_process is None:
                    # print('FFmpeg process is None.')
                    continue
                if self._ffmpeg_process.poll() is not None:
                    # print('FFmpeg process has exited.')
                    continue
                # print('Writing frame to FFmpeg stdin.')
                self._ffmpeg_process.stdin.write(frame.tobytes())
                # print('Frame written to FFmpeg stdin.')
            except Exception as e:
                print(f"Error writing to ffmpeg stdin: {e}")
        print('Live stream worker stopped.')

    def _stop_streaming(self) -> None:
        if self._ffmpeg_process is None:
            return
        try:
            if self._ffmpeg_process.stdin:
                self._ffmpeg_process.stdin.close()
        except Exception:
            pass
        try:
            self._ffmpeg_process.terminate()
        except Exception:
            pass
        self._ffmpeg_process = None

    def detect_worker(self):
        global detect_frame_count,trackid_saved_known
        trackid_saved_known = set()

        detect_frame_count = 0
        while self._running:
            if self._latest_frame is None:
                continue
            
            detect_frame_count += 1
            # print(detect_frame_count)
            # if detect_frame_count % 4 != 0:
            #     continue

            with self._frame_lock:
                frame_copy = self._latest_frame.copy()

            rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            # print("start face")
            faces = self._face_app.get(rgb)
            # print("end face")
            rects = []
            embeddings = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue
                
                
                                
                rects.append([x1, y1, x2, y2])
                embeddings.append(face.embedding)

            rects2 = []

            # TRACK EVERY FRAME
            # print(1)
            tracks = self._tracker.update(rects, classId="face")
            frame = frame_copy.copy()
            raw_frame = frame_copy.copy()
            for track in tracks:
                x1, y1, x2, y2, track_id, _ = track

                matched_embedding = None

                for rect, emb in zip(rects, embeddings):
                    rx1, ry1, rx2, ry2 = rect

                    if abs(x1 - rx1) < 15 and abs(y1 - ry1) < 15:
                        matched_embedding = emb
                        break

                if matched_embedding is None:
                    continue
                if  self._trackid_to_name.get(track_id) is None:
                    h, w = frame.shape[:2]
                    self._trackid_to_name[track_id] = {
                        "name": "Unknown",
                        "score": 0.0,
                        "student": None,
                        "frames": [],
                        "frame": None,
                        "attendance_id": None,
                        "video_writers": cv2.VideoWriter(os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4"), VIDEO_FOURCC, VIDEO_FPS, (w, h))
                    }
                tracker = self._trackid_to_name.get(track_id)
                
                
                    # print(2)
                name, score = self._recognize(matched_embedding)
                # print(track_id, name, score)
                    # print(3)
                    
                if tracker["name"] == "Unknown":
                    tracker["name"] = name
                else:
                    if name == "Unknown":
                        name = tracker["name"]
                    elif name != tracker["name"]:
                        continue
                    
                tracker["score"] = score
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"ID:{track_id} {name} {score:.2f}" \
                    if name != "Unknown" else f"ID:{track_id} Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                frame_tracker = frame_copy.copy()
                cv2.rectangle(frame_tracker, (x1, y1), (x2, y2), color, 2)
                
                tracker["video_writers"].write(frame_tracker)

                if name != "Unknown":
                    if track_id not in trackid_saved_known:
                        trackid_saved_known.add(track_id)
                        
                        studentRep = StudentRepository()
                        findStudent = studentRep.find({"_id": ObjectId(name)})
                        # print(findStudent)
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
                                tracker["attendance_id"] = self._register_attendance(student, score, raw_frame, ( x1, y1, x2, y2))
                    else:
                        student = tracker.get("student")
                        if student is not None:
                            label = f"ID:{track_id} {student.name} {score:.2f}"
                
                if os.path.exists(FONT_PATH):
                    frame = cv2_putText_utf8(frame, label, (x1, y1-40), FONT_PATH, 30, color)
                else:
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                #kiểm tra frame_detect nếu không tồn tại trong frames thì append vào tracker["frames"]
                if frame_copy is not None:
                    face_crop_avatar = frame_copy[
                        max(y1-int((abs(y1-y2))*0.2), 0):max(y2+int((abs(y1-y2))*0.2), 0),
                        max(x1-int((abs(x1-x2))*0.2), 0):max(x2+int((abs(x1-x2))*0.2), 0),
                    ]
                    face_crop_build = frame_copy[y1:y2, x1:x2]
                    if face_crop_build is None or face_crop_build.size == 0:
                        continue
                    is_blur, variance = check_blur_laplacian(face_crop_build)
                    # if(variance < 3):
                    #     qt_invoke(lambda: self._update_attendance_panel(Attendance(), face_crop_avatar))
                    # print(is_blur, variance)
                    # is_blur = False
                    if not is_blur:
                        if not any(detect_frame_count is c for _, c, f in tracker["frames"]):
                            tracker["frames"].append((variance, detect_frame_count, face_crop_build.copy()))
                        if score > tracker.get("score", 0):
                            tracker["score"] = score    
                            tracker["frame"] = face_crop_avatar.copy()
                        else:
                            if variance > tracker.get("variance", 0):
                                tracker["frame"] = face_crop_avatar.copy()
                                tracker["variance"] = variance
                        
                # lấy 15 frames có area lớn nhất
                tracker["frames"] = sorted(tracker["frames"], key=lambda x: x[0], reverse=True)[:15]
                rects2.append((track_id,color,label,[x1, y1, x2, y2]))
                # print(4)
                    

            self._latest_faces = rects2
            time.sleep(0.1)
            

    def _stop_capture(self) -> None:
        if not self._running and self._capture is None:
            return

        self._running = False
        self.open_btn.setText("Bắt đầu")
        self._timer.stop()

        for thread in (self._capture_thread, self._detect_thread, self._live_stream_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1)

        self._stop_streaming()

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
        # skip_frames = 3
        # frame_count = 0
        # last_seen_faces: List[Tuple[np.ndarray, str, float]] = []

        while self._running and self._capture is not None:
            try:
                ret, frame = self._capture.read()
            except cv2.error:
                break
            if not ret:
                time.sleep(0.01)
                continue

            if self.source_combo.currentText() == "Webcam":
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            with self._frame_lock:
                self._latest_frame = frame.copy()

        self._running = False

    def _recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        if self._annoy_index is None or self._annoy_index.get_n_items() == 0:
            # print("Annoy index is not initialized or empty.")
            return "Unknown", 0.0

        idx, dist = self._annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)
        if not dist:
            # print("No nearest neighbors found.")
            return "Unknown", 0.0

        sim = 1 - (dist[0] ** 2) / 2
        if sim >= SIM_THRESHOLD:
            # print(f"Recognized: {self._idx2name.get(str(idx[0]), "Unknown")} with similarity {sim:.2f}")
            return self._idx2name.get(str(idx[0]), "Unknown"), sim
        # print(f"Unrecognized with similarity {sim:.2f}")
        return "Unknown", sim

    def _register_attendance(self, student: Student, score: float, frame: np.ndarray, bbox: np.ndarray) -> None:
        now = time.time()
        last_time = self._last_seen.get(student.id, 0)
        if now - last_time < 10:
            return

        self._last_seen[student.id] = now
        timestamp = now_local()
        
        face_crop = frame[
            max(bbox[1]-int((abs(bbox[1]-bbox[3]))*0.2), 0):max(bbox[3]+int((abs(bbox[1]-bbox[3]))*0.2), 0),
            max(bbox[0]-int((abs(bbox[0]-bbox[2]))*0.2), 0):max(bbox[2]+int((abs(bbox[0]-bbox[2]))*0.2), 0),
        ]
        
        # path = os.path.join(
        #     CHECKIN_DIR,
        #     f"{student.id}_{now_local().strftime("%Y%m%d_%H%M%S")}.jpg"
        # )
        # self._save_queue.put((face_crop.copy(), path))
        
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
        # self._handle_disappeared()
        
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
            # print(target_size)
            
            if target_size.width() <= 0:
                target_size = QtCore.QSize(120, self.avatar.height())

            # print(target_size)
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
            
            # print(self.avatar.size())
            # tmp_label = QtWidgets.QLabel()
            # tmp_label.setPixmap(canvas)
            # tmp_label.show()
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

    def _render_frame(self) -> None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()

        if frame is None:
            return

        raw_frame = frame.copy()
        # rects, embeddings, (frame_count, frame_detect) = self._latest_faces if self._latest_faces else ([], [], (None, None))
        rects = self._latest_faces if self._latest_faces else []
        
        for track_id, color, label, (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if os.path.exists(FONT_PATH):
                frame = cv2_putText_utf8(frame, label, (x1, y1-40), FONT_PATH, 30, color)
            else:
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_frame.setPixmap(pixmap.scaled(
            self.video_frame.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        try:
            self._livestream_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def _show_history(self) -> None:
        self.history_table.scrollToBottom()
        
       

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        print("app closing")
        self._stop_capture()
        self._release_capture()
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    init_qt_invoker()
    window = AttendanceWindow()
    app.aboutToQuit.connect(window._stop_capture)
    
    window.show()
    # window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
