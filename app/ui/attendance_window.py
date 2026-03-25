from __future__ import annotations

import os
from typing import Optional

from bson import ObjectId
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from app.config import (
    AVATAR_DIR,
    CHECKIN_DIR,
    DEFAULT_RTSP_URL,
    VIDEO,
)
from app.ui.dialogs import UpdateStudentDialog
from app.ui.styles import STYLE_SHEET
from app.utils.mongodb_access import (
    Attendance,
    Student,
    to_local,
)
from app.utils.qt_invoker import qt_invoke

from scripts.services.attendance import AttendanceService
from scripts.utils.image_utils import get_student_avatar_path


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

        # -- attendance service (all logic lives here) --
        self._svc = AttendanceService()
        self._svc.on_status = lambda msg: qt_invoke(lambda: self.status_label.setText(msg))
        self._svc.on_attendance = lambda rec: qt_invoke(lambda: self._on_new_attendance(rec))
        self._svc.tracker.on_disappeared = lambda tid: qt_invoke(lambda: self._svc.handle_disappeared(tid))
        self._svc.on_video_end = lambda: qt_invoke(lambda: self._on_video_end())

        self._attendance_selected: Optional[Attendance] = None
        self._attendance_selected_row: Optional[int] = None

        self._svc.load_recognition_assets()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._render_frame)

        self.setStyleSheet(STYLE_SHEET)
        self.load_params()

    # ==================================================================
    # Recognition helpers (delegate to service)
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

        list_attendance = self._svc.load_today_attendance()

        self.clear_selected_row()
        while self.history_table.rowCount() > 0:
            self.history_table.removeRow(0)

        for attendance in list_attendance:
            self._append_history_row(attendance)

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
        self._svc.build_face()

    def _check_things(self) -> None:
        print("Checking things...")
        self._svc.dump_tracks()

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

    def _on_new_attendance(self, record: Attendance) -> None:
        self._update_attendance_panel(record)
        self._append_history_row(record)
        
    def _on_video_end(self) -> None:
        qt_invoke(lambda: self._stop_capture())
        QtWidgets.QMessageBox.information(self, "Thông báo", "Video đã kết thúc.")

    # ==================================================================
    # Update student dialog
    # ==================================================================

    def _open_update_student_dialog(self) -> None:
        if self._attendance_selected is None:
            QtWidgets.QMessageBox.information(self, "Thiếu dữ liệu", "Chưa chọn học sinh để cập nhật.")
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

        # A new classroom may have been created inside the dialog
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
                QtWidgets.QMessageBox.warning(self, "Thiếu RTSP", "Vui lòng nhập RTSP URL.")
                return
        elif self.source_combo.currentText() == "Video File":
            source = VIDEO

        if not self._svc.start_capture(source, self.source_combo.currentText()):
            QtWidgets.QMessageBox.critical(self, "Lỗi", "Không mở được nguồn video.")
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
    # Attendance panel (Qt-specific rendering)
    # ==================================================================

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
            face_crop = AttendanceService.load_face_crop_for_record(record)

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
    # Render (Qt-specific: convert cv2 frame -> QPixmap)
    # ==================================================================

    def _render_frame(self) -> None:
        frame = self._svc.build_annotated_frame()
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_frame.setPixmap(pixmap.scaled(
            self.video_frame.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))

    # ==================================================================
    # Close
    # ==================================================================

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        print("app closing")
        self._svc.close()
        event.accept()
