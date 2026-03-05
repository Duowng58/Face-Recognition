from enum import unique
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from bson import ObjectId
import cv2
from matplotlib.animation import writers
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


from insightface.app import FaceAnalysis
from annoy import AnnoyIndex



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
from checkin.face_tracker import Tracker
from utils.qt_invoker import init_qt_invoker, qt_invoke


FACE_DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, "face_data_1"))
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH = os.path.join(FACE_DATA_DIR, "image_paths.json")

MIN_BBOX_AREA = 10000
EMBEDDING_DIM = 512
TREE = 50
SIM_THRESHOLD = 0.6
CAPTURE_ROOT = "captured_faces"
KNOWN_DIR = os.path.join(CAPTURE_ROOT, "known")
CHECKIN_DIR = os.path.join(CAPTURE_ROOT, "checkin")
UNKNOWN_DIR = os.path.join(CAPTURE_ROOT, "unknown")
VIDEO_FPS    = 15
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
DEFAULT_RTSP_URL = "rtsp://admin:Ancovn1234@192.168.1.64:554/Streaming/Channels/201/video"
FONT_PATH = os.path.join(ROOT_DIR, "app", "fonts", "Arial.ttf")



def build_placeholder_pixmap(size: QtCore.QSize, label: str) -> QtGui.QPixmap:
    pixmap = QtGui.QPixmap(size)
    pixmap.fill(QtGui.QColor("#f3f5f7"))

    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    rect = QtCore.QRect(0, 0, size.width(), size.height())
    painter.setPen(QtGui.QPen(QtGui.QColor("#d0d5da"), 2))
    painter.drawRect(rect.adjusted(10, 10, -10, -10))

   

    painter.end()

    return pixmap


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
        self._video_write_queue = queue.Queue()

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
        print(f"total frames: {len(tracker["frames"])}, "
                f"name: {tracker.get("name", "--")}, "
                f"variance: {tracker.get("variance", 0)}, "
                f"score: {tracker.get("score", 0)}, "
                f"student: {tracker.get("student", "--")}")
        time = now_local()
        def save_frames(attendance_id):
            now_path = os.path.join(CHECKIN_DIR, time.strftime("%Y-%m-%d"), str(attendance_id))
            self._save_queue.put((tracker.get("frame"), os.path.join(now_path, "frame.jpg")))
            if len(tracker["frames"]) > 0:
                for index, (area, frame_count, frame) in enumerate(tracker["frames"]):
                    self._save_queue.put((frame, os.path.join(now_path, "frames", f"frame_{index}.jpg")))
                    
            if tracker["video_writers"] is not None:
                tracker["video_writers"].release()
                old_path = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                new_path = os.path.join(CHECKIN_DIR, time.strftime("%Y-%m-%d"), str(attendance_id), f"video.mp4")
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                os.rename(old_path, new_path)
        
        if tracker.get("name") == "Unknown":
            if tracker.get("frame") is None:
                return
            if len(tracker["frames"]) < 10:
                return
            print("Process unknown tracker")
            # Handle unknown tracker
            attendanceRepo = AttendanceRepository()
            attendance_unknown_id = ObjectId()
            # Tạo folder 
            save_frames(attendance_unknown_id)
                    
            
            attendance_unknown = Attendance(
                id= attendance_unknown_id,
                time=time,
                student_name="Unknown",
                score=tracker.get("score", 0),
            )
            attendanceRepo.insert(attendance_unknown)
            self.build_unknown_face(tracker,attendance_unknown)
            
            qt_invoke(lambda: self._append_history_row(attendance_unknown))
        else:
            if tracker["attendance_id"] is not None:
                save_frames(tracker["attendance_id"])
            else:
                if tracker["video_writers"] is not None:
                    tracker["video_writers"].release()
                    old_path = os.path.join(CHECKIN_DIR, f"tmp_video_{track_id}.mp4")
                    os.remove(old_path)

                    
        self._trackid_to_name.pop(track_id, None)
        
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
        client = MongoClientSingleton.get_client()
        self.list_classrooms = list(client.db["classrooms"].find())
        print(self.list_classrooms)
        
        attendanceRepo = AttendanceRepository()
        list_attendance = attendanceRepo.find({
            "time": {"$gte": start_of_today_local()}
        })
        for attendance in list_attendance:
            self._append_history_row(attendance)
        

    def _build_form_row(self) -> QtWidgets.QLayout:
        row = QtWidgets.QGridLayout()
        row.setHorizontalSpacing(12)
        row.setVerticalSpacing(6)

        source_label = QtWidgets.QLabel("Nguồn video")
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP"])

        rtsp_label = QtWidgets.QLabel("RTSP URL")
        self.rtsp_edit = QtWidgets.QLineEdit(DEFAULT_RTSP_URL)

        row.addWidget(source_label, 0, 0)
        row.addWidget(self.source_combo, 0, 1)
        row.addWidget(rtsp_label, 0, 2)
        row.addWidget(self.rtsp_edit, 0, 3, 1, 3)

        row.setColumnStretch(1, 1)
        row.setColumnStretch(3, 1)
        row.setColumnStretch(5, 1)
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
            print(f"total frames: {len(tracker["frames"])}, "
                  f"name: {tracker.get("name", "--")}, "
                  f"area: {tracker.get("area", 0)}, "
                  f"score: {tracker.get("score", 0)}, "
                  f"student: {tracker.get("student", "--")}")
        
    def _build_info_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QGroupBox()
        # panel.setFixedWidth(600)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(10)
        
        self.success_label = QtWidgets.QLabel("Thông tin học sinh")
        self.success_label.setObjectName("successTitle")
        layout.addWidget(self.success_label)

        self.avatar = QtWidgets.QLabel()
        self.avatar.setFixedHeight(180)
        self.avatar.setObjectName("avatar")
        self.avatar.setAlignment(QtCore.Qt.AlignCenter)
        # self.avatar.setPixmap(build_placeholder_pixmap(QtCore.QSize(200, 180), "?"))
        # self.avatar.setScaledContents(True)
        layout.addWidget(self.avatar)

        info_form = QtWidgets.QFormLayout()
        info_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        info_form.setFormAlignment(QtCore.Qt.AlignTop)

        self.id_value = QtWidgets.QLabel("--")
        self.name_value = QtWidgets.QLabel("--")
        self.class_value = QtWidgets.QLabel("--")
        self.time_value = QtWidgets.QLabel("--")
        info_form.addRow("ID Sinh Viên:", self.id_value)
        info_form.addRow("Tên Sinh Viên:", self.name_value)
        info_form.addRow("Lớp:", self.class_value)
        info_form.addRow("Thời gian:", self.time_value)
        layout.addLayout(info_form)
        
        
        

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
        attendance = self.history_table.item(row, 0).data(QtCore.Qt.UserRole)
        if attendance is not None:
            self._update_attendance_panel(attendance)
            print(attendance)


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
                background: #2a2a2a;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                padding: 6px 10px;
                color: #e5e7eb;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
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
        self._running = True
        self.open_btn.setText("Kết thúc")
        
        # self.latest_frame = None
        self._latest_faces = []
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        self._detect_thread = threading.Thread(target=self.detect_worker, daemon=True)
        self._detect_thread.start()
        
        self._timer.start()
        

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

        for thread in (self._capture_thread, self._detect_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1)

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
            frame = cv2.resize(frame, (1080, 720))

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
            f"Thông báo: {record.student_name} đã điểm danh (score {record.score:.2f})."
        )
        self.status_label.setText(status)
        
        if face_crop is None:
            now_path = os.path.join(CHECKIN_DIR, time.strftime("%Y-%m-%d"), str(record.id))
            print(now_path)
            if os.path.exists(os.path.join(now_path, "frame.jpg")):
                print(os.path.join(now_path, "frame.jpg"))
                face_crop = cv2.imread(os.path.normpath(os.path.join(now_path, "frame.jpg")))


        if face_crop is not None and face_crop.size > 0:
            target_size = self.avatar.size()
            # print(target_size)
            
            if target_size.width() <= 0:
                target_size = QtCore.QSize(200, self.avatar.height())

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
