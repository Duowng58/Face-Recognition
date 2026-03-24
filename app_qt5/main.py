"""
PyQt5 version of the Face-Recognition attendance application.

Single-file app that reuses ``scripts.services.attendance.AttendanceService``
for all non-UI logic (capture, detection, tracking, recognition, DB).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# ── path setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataclasses import dataclass

from bson import ObjectId
import cv2
import numpy as np

# ── IMPORTANT: import scripts (onnxruntime / insightface) BEFORE PyQt5 ──
# PyQt5 ships its own Qt5 DLLs that can conflict with onnxruntime's
# native library loading on Windows. Importing onnxruntime first avoids
# the "DLL initialization routine failed" crash.
from scripts.config import (
    AVATAR_DIR,
    CHECKIN_DIR,
    DEFAULT_RTSP_URL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    VIDEO,
)
from scripts.services.attendance import AttendanceService
from scripts.utils.image_utils import (
    get_attendance_frame_path,
    get_student_avatar_path,
)
from scripts.utils.mongodb_access import (
    Attendance,
    MongoClientSingleton,
    Student,
    StudentRepository,
    to_local,
)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QSize, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QCloseEvent, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
)


# ══════════════════════════════════════════════════════════════
# Qt-invoke helper  (thread-safe callback dispatch to UI thread)
# ══════════════════════════════════════════════════════════════

class _QtInvoker(QObject):
    """Dispatch any callable to the main/UI thread via signal/slot."""

    _signal = pyqtSignal(object, int)

    def __init__(self) -> None:
        super().__init__()
        self._signal.connect(self._on_invoke)

    def _on_invoke(self, cb, delay_ms: int) -> None:
        if delay_ms and delay_ms > 0:
            QTimer.singleShot(delay_ms, cb)
        else:
            cb()


_invoker: Optional[_QtInvoker] = None


def init_qt_invoker() -> None:
    global _invoker
    if _invoker is None:
        _invoker = _QtInvoker()


def qt_invoke(cb, delay_ms: int = 0) -> None:
    """Schedule *cb* on the main thread."""
    if _invoker is not None:
        _invoker._signal.emit(cb, delay_ms)


# ══════════════════════════════════════════════════════════════
# Style Sheet  (dark theme, matching the PySide6 version)
# ══════════════════════════════════════════════════════════════

STYLE_SHEET = """
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
#secondaryButton {
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


# ══════════════════════════════════════════════════════════════
# Student Picker Dialog
# ══════════════════════════════════════════════════════════════

class StudentPickerDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        students: list[Student],
        classrooms: list[dict],
        avatar_dir: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chọn học sinh")
        self.setModal(True)

        self._students = students
        self._class_map = {str(c.get("_id")): c.get("name", "") for c in classrooms}
        self._avatar_dir = avatar_dir

        layout = QVBoxLayout(self)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["ID", "Lớp", "Tên", "Ảnh"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._populate_table()
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._table.itemDoubleClicked.connect(lambda _: self.accept())

    def _populate_table(self) -> None:
        for student in self._students:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(student.id)))
            self._table.setItem(row, 1, QTableWidgetItem(self._class_map.get(str(student.class_id), "")))
            self._table.setItem(row, 2, QTableWidgetItem(student.name))

            image_item = QTableWidgetItem("--")
            image_path = get_student_avatar_path(self._avatar_dir, student.id)
            if image_path:
                pixmap = QPixmap(image_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_item = QTableWidgetItem()
                image_item.setIcon(QIcon(pixmap))
                image_item.setSizeHint(QSize(120, 120))
            self._table.setItem(row, 3, image_item)
            self._table.item(row, 0).setData(Qt.UserRole, student)

    def selected_student(self) -> Optional[Student]:
        row = self._table.currentRow()
        if row < 0:
            return None
        return self._table.item(row, 0).data(Qt.UserRole)

    @staticmethod
    def pick(parent, students, classrooms, avatar_dir) -> Optional[Student]:
        dlg = StudentPickerDialog(parent, students, classrooms, avatar_dir)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.selected_student()


# ══════════════════════════════════════════════════════════════
# Update Student Dialog
# ══════════════════════════════════════════════════════════════

@dataclass
class UpdateStudentResult:
    new_name: str
    new_class_id: Optional[ObjectId]
    class_name: str
    selected_student_id: Optional[ObjectId]
    attendance_avatar_pixmap: Optional[QPixmap]


class UpdateStudentDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        attendance: Attendance,
        student: Optional[Student],
        student_id: Optional[ObjectId],
        classrooms: list[dict],
        avatar_dir: str,
        checkin_dir: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cập nhật học sinh")
        self.setModal(True)

        self._attendance = attendance
        self._classrooms = classrooms
        self._avatar_dir = avatar_dir
        self._checkin_dir = checkin_dir
        self._student_id = student_id
        self._selected_student: Optional[Student] = None

        student_repo = StudentRepository()
        self._all_students = student_repo.find()

        self.result: Optional[UpdateStudentResult] = None

        form = QFormLayout(self)

        # -- avatar images --
        student_avatar_box = QWidget()
        sa_layout = QVBoxLayout(student_avatar_box)
        sa_layout.setContentsMargins(0, 0, 0, 0)
        sa_layout.addWidget(QLabel("Ảnh học sinh"))
        self._student_avatar = QLabel()
        self._student_avatar.setObjectName("avatar")
        self._student_avatar.setFixedSize(180, 180)
        self._student_avatar.setAlignment(Qt.AlignCenter)
        sa_layout.addWidget(self._student_avatar)

        attendance_avatar_box = QWidget()
        aa_layout = QVBoxLayout(attendance_avatar_box)
        aa_layout.setContentsMargins(0, 0, 0, 0)
        aa_layout.addWidget(QLabel("Ảnh điểm danh"))
        self._attendance_avatar = QLabel()
        self._attendance_avatar.setObjectName("avatar")
        self._attendance_avatar.setFixedSize(180, 180)
        self._attendance_avatar.setAlignment(Qt.AlignCenter)
        aa_layout.addWidget(self._attendance_avatar)

        images_row = QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.addWidget(student_avatar_box)
        images_row.addWidget(attendance_avatar_box)

        self._refresh_student_avatar(student_id)
        self._refresh_attendance_avatar(attendance.id)

        # -- id / picker --
        self._id_edit = QLineEdit(str(student.id) if student else "--")
        self._id_edit.setStyleSheet("QLineEdit { background:#2a2a2a; color: gray; }")
        self._id_edit.setReadOnly(True)
        self._name_edit = QLineEdit(student.name if student else attendance.student_name)

        select_btn = QPushButton("Chọn")
        select_btn.setObjectName("primaryButton")
        select_btn.clicked.connect(self._handle_student_pick)

        # -- class combo --
        self._class_combo = QComboBox()
        self._class_combo.addItem("Chọn lớp...", None)
        for c in classrooms:
            self._class_combo.addItem(c.get("name", ""), c.get("_id"))
        self._class_combo.addItem("+ Thêm lớp mới", "__new__")

        self._new_class_edit = QLineEdit()
        self._new_class_edit.setPlaceholderText("Nhập tên lớp mới")
        self._new_class_edit.setVisible(False)

        if student and student.class_id is not None:
            self._set_class_selection(student.class_id)

        self._class_combo.currentIndexChanged.connect(self._handle_class_change)
        self._handle_class_change()

        picker_row = QHBoxLayout()
        picker_row.addWidget(self._id_edit, stretch=1)
        picker_row.addWidget(select_btn)

        form.addRow("", images_row)
        form.addRow("ID:", picker_row)
        form.addRow("Tên:", self._name_edit)

        class_row = QVBoxLayout()
        class_row.addWidget(self._class_combo)
        class_row.addWidget(self._new_class_edit)
        form.addRow("Lớp:     ", class_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    # helpers
    @staticmethod
    def _load_avatar(path: str) -> QPixmap:
        if not path or not os.path.exists(path):
            return QPixmap()
        return QPixmap(path).scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _refresh_student_avatar(self, sid) -> None:
        path = get_student_avatar_path(self._avatar_dir, sid)
        if path and os.path.exists(path):
            self._student_avatar.setPixmap(self._load_avatar(path))
        else:
            self._student_avatar.clear()

    def _refresh_attendance_avatar(self, aid) -> None:
        path = get_attendance_frame_path(self._checkin_dir, self._attendance.time, aid)
        if path and os.path.exists(path):
            self._attendance_avatar.setPixmap(self._load_avatar(path))
        else:
            self._attendance_avatar.clear()

    def _set_class_selection(self, class_id) -> None:
        if class_id is None:
            return
        for idx in range(self._class_combo.count()):
            if str(self._class_combo.itemData(idx)) == str(class_id):
                self._class_combo.setCurrentIndex(idx)
                break

    # slots
    def _handle_student_pick(self) -> None:
        picked = StudentPickerDialog.pick(
            self, self._all_students, self._classrooms, self._avatar_dir,
        )
        if picked is None:
            return
        self._selected_student = picked
        self._name_edit.setText(picked.name)
        self._set_class_selection(picked.class_id)
        self._refresh_student_avatar(picked.id)

    def _handle_class_change(self) -> None:
        is_new = self._class_combo.currentData() == "__new__"
        self._new_class_edit.setVisible(is_new)

    def _on_accept(self) -> None:
        new_name = self._name_edit.text().strip()
        selected_student_id = self._selected_student.id if self._selected_student else None
        new_class_id = self._class_combo.currentData()

        if not new_name:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên học sinh.")
            return

        if new_class_id == "__new__":
            new_class_name = self._new_class_edit.text().strip()
            if not new_class_name:
                QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên lớp mới.")
                return
            client = MongoClientSingleton.get_client()
            ins = client.db["classrooms"].insert_one({"name": new_class_name})
            new_class_id = ins.inserted_id
            class_name = new_class_name
        elif new_class_id is None:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng chọn lớp.")
            return
        else:
            class_name = self._class_combo.currentText()

        self.result = UpdateStudentResult(
            new_name=new_name,
            new_class_id=new_class_id,
            class_name=class_name,
            selected_student_id=selected_student_id,
            attendance_avatar_pixmap=self._attendance_avatar.pixmap(),
        )
        self.accept()

    @staticmethod
    def show(parent, attendance, student, student_id, classrooms, avatar_dir, checkin_dir):
        dlg = UpdateStudentDialog(
            parent, attendance, student, student_id,
            classrooms, avatar_dir, checkin_dir,
        )
        if dlg.exec_() != QDialog.Accepted:
            return None
        return dlg.result


# ══════════════════════════════════════════════════════════════
# Attendance Window  (main window)
# ══════════════════════════════════════════════════════════════

class AttendanceWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hệ thống điểm danh khuôn mặt")
        self.setMinimumSize(1180, 720)

        self._central = QWidget()
        self.setCentralWidget(self._central)

        root = QVBoxLayout(self._central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Hệ thống điểm danh khuôn mặt")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        root.addLayout(self._build_form_row())

        content = QHBoxLayout()
        content.setSpacing(14)
        root.addLayout(content)

        content.addWidget(self._build_video_panel(), stretch=3)
        content.addWidget(self._build_info_panel(), stretch=2)

        # -- service --
        self._svc = AttendanceService()
        self._svc.on_status = lambda msg: qt_invoke(lambda: self.status_label.setText(msg))
        self._svc.on_attendance = lambda rec: qt_invoke(lambda: self._on_new_attendance(rec))
        self._svc.tracker.on_disappeared = lambda tid: qt_invoke(lambda: self._svc.handle_disappeared(tid))
        self._svc.on_video_end = lambda: qt_invoke(lambda: self._on_video_end())

        self._attendance_selected: Optional[Attendance] = None
        self._attendance_selected_row: Optional[int] = None

        self._svc.load_recognition_assets()

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._render_frame)

        self.setStyleSheet(STYLE_SHEET)
        self.load_params()

    # ==================================================================
    # Recognition helpers
    # ==================================================================

    def _load_recognition_assets(self) -> None:
        self._svc.load_recognition_assets()

    def _build_face(self) -> None:
        self._svc.build_face()

    # ==================================================================
    # Data loading
    # ==================================================================

    def load_params(self) -> None:
        self._svc.refresh_classrooms()
        records = self._svc.load_today_attendance()

        self.clear_selected_row()
        while self.history_table.rowCount() > 0:
            self.history_table.removeRow(0)

        for rec in records:
            self._append_history_row(rec)

    def clear_selected_row(self) -> None:
        self._attendance_selected = None
        self.id_value.setText("--")
        self.name_value.setText("--")
        self.class_value.setText("--")
        self.time_value.setText("--")
        self.status_label.setText("-")
        self.avatar.clear()
        self._update_student_image(None)

    # ==================================================================
    # UI builders
    # ==================================================================

    def _build_form_row(self) -> QGridLayout:
        row = QGridLayout()
        row.setHorizontalSpacing(12)
        row.setVerticalSpacing(6)

        source_label = QLabel("Nguồn video")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam", "RTSP", "Video File"])

        rtsp_label = QLabel("RTSP URL")
        self.rtsp_edit = QLineEdit(DEFAULT_RTSP_URL)

        self.stream_toggle = QCheckBox("Livestream")
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

    def _build_video_panel(self) -> QGroupBox:
        panel = QGroupBox()
        lay = QVBoxLayout(panel)
        lay.setSpacing(10)

        self.video_frame = QLabel()
        self.video_frame.setObjectName("videoFrame")
        self.video_frame.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.video_frame, stretch=1)

        status_box = QFrame()
        status_box.setObjectName("statusBox")
        sl = QHBoxLayout(status_box)
        sl.setContentsMargins(12, 8, 12, 8)
        self.status_label = QLabel("Thông báo: Chưa có dữ liệu điểm danh.")
        sl.addWidget(self.status_label)
        lay.addWidget(status_box)

        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Bắt đầu")
        self.open_btn.setObjectName("primaryButton")
        self.open_btn.clicked.connect(self._toggle_capture)

        self.build_face_btn = QPushButton("Nhập dữ liệu học sinh")
        self.build_face_btn.setObjectName("secondaryButton")
        self.build_face_btn.clicked.connect(self._import_data)

        self.history_btn = QPushButton("Check")
        self.history_btn.setObjectName("ghostButton")
        self.history_btn.clicked.connect(self._check_things)

        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.build_face_btn)
        btn_row.addWidget(self.history_btn)
        lay.addLayout(btn_row)
        return panel

    def _build_info_panel(self) -> QGroupBox:
        panel = QGroupBox()
        lay = QVBoxLayout(panel)
        lay.setSpacing(10)

        self.success_label = QLabel("Thông tin điểm danh")
        self.success_label.setObjectName("successTitle")
        lay.addWidget(self.success_label)

        btn_layout = QHBoxLayout()
        avatar_col = QHBoxLayout()
        avatar_col.setSpacing(6)

        self.avatar = QLabel()
        self.avatar.setFixedSize(120, 120)
        self.avatar.setObjectName("avatar")
        self.avatar.setAlignment(Qt.AlignCenter)
        avatar_col.addWidget(self.avatar)

        self.student_image = QLabel()
        self.student_image.setFixedSize(120, 120)
        self.student_image.setObjectName("avatar")
        self.student_image.setAlignment(Qt.AlignCenter)
        avatar_col.addWidget(self.student_image)
        avatar_col.addStretch(1)
        btn_layout.addLayout(avatar_col, stretch=1)

        action_col = QVBoxLayout()
        self.update_student_btn = QPushButton("Cập nhật")
        self.update_student_btn.setObjectName("ghostButton")
        self.update_student_btn.clicked.connect(self._open_update_student_dialog)
        action_col.addWidget(self.update_student_btn, alignment=Qt.AlignTop)
        btn_layout.addLayout(action_col)
        lay.addLayout(btn_layout)

        info_form = QFormLayout()
        info_form.setLabelAlignment(Qt.AlignLeft)
        info_form.setFormAlignment(Qt.AlignTop)

        self.id_value = QLabel("--")
        self.name_value = QLabel("--")
        self.class_value = QLabel("--")
        self.time_value = QLabel("--")
        info_form.addRow("ID Học sinh:", self.id_value)
        info_form.addRow("Tên Học sinh:", self.name_value)
        info_form.addRow("Lớp:", self.class_value)
        info_form.addRow("Thời gian:", self.time_value)
        lay.addLayout(info_form)

        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setObjectName("hLine")
        lay.addWidget(divider)

        history_label = QLabel("Lịch sử điểm danh")
        history_label.setObjectName("historyTitle")
        lay.addWidget(history_label)

        self.history_table = QTableWidget(0, 4)
        self.history_table.cellClicked.connect(self._on_history_cell_clicked)
        self.history_table.setHorizontalHeaderLabels(["Lớp", "Tên học sinh", "Thời gian", "Score"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setMinimumHeight(180)
        lay.addWidget(self.history_table, stretch=1)

        return panel

    def _import_data(self) -> None:
        print("Importing data...")
        self.load_params()
        self._svc.build_face()

    def _check_things(self) -> None:
        print("Checking things...")
        self._svc.dump_tracks()

    # ==================================================================
    # History table
    # ==================================================================

    def _on_history_cell_clicked(self, row: int, column: int) -> None:
        self._attendance_selected = self.history_table.item(row, 0).data(Qt.UserRole)
        self._attendance_selected_row = row
        if self._attendance_selected is not None:
            self._update_attendance_panel(self._attendance_selected)

    def _append_history_row(self, record: Attendance) -> None:
        row = 0
        self.history_table.insertRow(row)
        classroom_item = QTableWidgetItem(record.student_classroom)
        classroom_item.setData(Qt.UserRole, record)
        self.history_table.setItem(row, 0, classroom_item)
        self.history_table.setItem(row, 1, QTableWidgetItem(record.student_name))
        self.history_table.setItem(
            row, 2,
            QTableWidgetItem(to_local(record.time).strftime("%H:%M:%S")),
        )
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{record.score:.2f}"))
        self.history_table.scrollToTop()

    def _on_new_attendance(self, record: Attendance) -> None:
        self._update_attendance_panel(record)
        self._append_history_row(record)

    def _on_video_end(self) -> None:
        qt_invoke(lambda: self._stop_capture())
        QMessageBox.information(self, "Thông báo", "Video đã kết thúc.")

    # ==================================================================
    # Update student dialog
    # ==================================================================

    def _open_update_student_dialog(self) -> None:
        if self._attendance_selected is None:
            QMessageBox.information(self, "Thiếu dữ liệu", "Chưa chọn học sinh để cập nhật.")
            return

        student = None
        student_id = self._attendance_selected.student_id
        if student_id is not None:
            try:
                if not isinstance(student_id, ObjectId):
                    student_id = ObjectId(str(student_id))
                student = self._svc.student_repo.get(student_id)
            except Exception:
                student_id = None

        result = UpdateStudentDialog.show(
            parent=self,
            attendance=self._attendance_selected,
            student=student,
            student_id=student_id,
            classrooms=self._svc.list_classrooms,
            avatar_dir=AVATAR_DIR,
            checkin_dir=CHECKIN_DIR,
        )
        if result is None:
            return

        self._svc.refresh_classrooms()

        if result.selected_student_id is not None:
            student_id = result.selected_student_id

        if student_id is None:
            student = Student(
                id=self._attendance_selected.id,
                name=result.new_name,
                class_id=result.new_class_id,
            )
            student_id = self._svc.student_repo.insert(student)
            os.makedirs(AVATAR_DIR, exist_ok=True)
            if result.attendance_avatar_pixmap is not None:
                result.attendance_avatar_pixmap.save(
                    os.path.join(AVATAR_DIR, f"{student_id}.jpg")
                )
        else:
            self._svc.student_repo.update(student_id, {
                "name": result.new_name,
                "class_id": result.new_class_id,
            })

        self._svc.attendance_repo.update(self._attendance_selected.id, {
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
    # Capture / Video  (delegate to AttendanceService)
    # ==================================================================

    def _toggle_capture(self) -> None:
        if self._svc.is_running:
            qt_invoke(self._stop_capture)
        else:
            qt_invoke(self._start_capture)

    def _start_capture(self) -> None:
        if self._svc.is_running:
            return

        source: int | str = 0
        if self.source_combo.currentText() == "RTSP":
            source = self.rtsp_edit.text().strip()
            if not source:
                QMessageBox.warning(self, "Thiếu RTSP", "Vui lòng nhập RTSP URL.")
                return
        elif self.source_combo.currentText() == "Video File":
            source = VIDEO

        if not self._svc.start_capture(source, self.source_combo.currentText()):
            QMessageBox.critical(self, "Lỗi", "Không mở được nguồn video.")
            return

        self._Frame_FPS = self._svc.frame_fps
        self.open_btn.setText("Kết thúc")
        self._timer.start()

    def _toggle_streaming(self, enabled: bool) -> None:
        self._svc.toggle_streaming(enabled)

    def _stop_capture(self) -> None:
        self._svc.stop_capture()
        self.open_btn.setText("Bắt đầu")
        self._timer.stop()
        self.video_frame.clear()

    # ==================================================================
    # Attendance panel
    # ==================================================================

    def _update_attendance_panel(self, record: Attendance, face_crop: np.ndarray = None) -> None:
        self.id_value.setText(str(record.student_id))
        self.name_value.setText(record.student_name)
        self.class_value.setText(record.student_classroom)
        self.time_value.setText(to_local(record.time).strftime("%H:%M:%S"))

        status = f"Thông báo: {record.student_name} đã điểm danh lúc {to_local(record.time).strftime('%H:%M:%S')}."
        self.status_label.setText(status)

        if face_crop is None:
            face_crop = AttendanceService.load_face_crop_for_record(record)

        if face_crop is not None and face_crop.size > 0:
            target_size = self.avatar.size()
            if target_size.width() <= 0:
                target_size = QSize(120, self.avatar.height())

            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bpl = ch * w
            qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.avatar.setPixmap(scaled)
        else:
            self.avatar.clear()

        self._update_student_image(record.student_id)

    def _update_student_image(self, student_id: Optional[ObjectId]) -> None:
        if student_id is None:
            self.student_image.clear()
            return

        path = get_student_avatar_path(AVATAR_DIR, student_id)
        if not path or not os.path.exists(path):
            self.student_image.clear()
            return

        target = self.student_image.size()
        if target.width() <= 0:
            target = QSize(120, self.student_image.height())

        pixmap = QPixmap(path).scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.student_image.setPixmap(pixmap)

    # ==================================================================
    # Render (convert BGR frame → QPixmap)
    # ==================================================================

    def _render_frame(self) -> None:
        frame = self._svc.build_annotated_frame()
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        qimg = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.video_frame.setPixmap(
            pixmap.scaled(self.video_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ==================================================================
    # Close
    # ==================================================================

    def closeEvent(self, event: QCloseEvent) -> None:
        print("app_qt5 closing")
        self._svc.close()
        event.accept()


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    init_qt_invoker()
    window = AttendanceWindow()
    app.aboutToQuit.connect(window._stop_capture)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
