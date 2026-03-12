from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from bson import ObjectId
from PySide6 import QtCore, QtGui, QtWidgets

from app.utils.image_utils import get_attendance_frame_path, get_student_avatar_path
from app.utils.mongodb_access import (
    Attendance,
    MongoClientSingleton,
    Student,
    StudentRepository,
)


# ======================================================================
# Student Picker Dialog
# ======================================================================


class StudentPickerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: QtWidgets.QWidget,
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

        layout = QtWidgets.QVBoxLayout(self)
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["ID", "Lớp", "Tên", "Ảnh"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        self._populate_table()

        layout.addWidget(self._table)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._table.itemDoubleClicked.connect(lambda _: self.accept())

    def _populate_table(self) -> None:
        for student in self._students:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(student.id)))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._class_map.get(str(student.class_id), "")))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(student.name))

            image_item = QtWidgets.QTableWidgetItem("--")
            image_path = get_student_avatar_path(self._avatar_dir, student.id)
            if image_path:
                pixmap = QtGui.QPixmap(image_path).scaled(120, 120, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                image_item = QtWidgets.QTableWidgetItem()
                image_item.setIcon(QtGui.QIcon(pixmap))
                image_item.setSizeHint(QtCore.QSize(120, 120))
            self._table.setItem(row, 3, image_item)
            self._table.item(row, 0).setData(QtCore.Qt.UserRole, student)

    def selected_student(self) -> Optional[Student]:
        current_row = self._table.currentRow()
        if current_row < 0:
            return None
        return self._table.item(current_row, 0).data(QtCore.Qt.UserRole)

    @staticmethod
    def pick(
        parent: QtWidgets.QWidget,
        students: list[Student],
        classrooms: list[dict],
        avatar_dir: str,
    ) -> Optional[Student]:
        dialog = StudentPickerDialog(parent, students, classrooms, avatar_dir)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dialog.selected_student()


# ======================================================================
# Update Student Dialog — result dataclass
# ======================================================================


@dataclass
class UpdateStudentResult:
    """Data returned by UpdateStudentDialog on accept."""
    new_name: str
    new_class_id: ObjectId | None
    class_name: str
    selected_student_id: ObjectId | None  # non-None when an existing student was picked
    attendance_avatar_pixmap: Optional[QtGui.QPixmap]  # for saving as new student avatar


# ======================================================================
# Update Student Dialog
# ======================================================================


class UpdateStudentDialog(QtWidgets.QDialog):
    """Dialog for updating the student linked to an attendance record."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
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

        # -- result populated in _on_accept --
        self.result: Optional[UpdateStudentResult] = None

        # -- build form --
        form = QtWidgets.QFormLayout(self)

        # avatars row
        student_avatar_box = QtWidgets.QWidget()
        student_avatar_box_layout = QtWidgets.QVBoxLayout(student_avatar_box)
        student_avatar_box_layout.setContentsMargins(0, 0, 0, 0)
        student_avatar_box_layout.addWidget(QtWidgets.QLabel("Ảnh học sinh"))
        self._student_avatar = QtWidgets.QLabel()
        self._student_avatar.setObjectName("avatar")
        self._student_avatar.setFixedSize(180, 180)
        self._student_avatar.setAlignment(QtCore.Qt.AlignCenter)
        student_avatar_box_layout.addWidget(self._student_avatar)

        attendance_avatar_box = QtWidgets.QWidget()
        attendance_avatar_box_layout = QtWidgets.QVBoxLayout(attendance_avatar_box)
        attendance_avatar_box_layout.setContentsMargins(0, 0, 0, 0)
        attendance_avatar_box_layout.addWidget(QtWidgets.QLabel("Ảnh điểm danh"))
        self._attendance_avatar = QtWidgets.QLabel()
        self._attendance_avatar.setObjectName("avatar")
        self._attendance_avatar.setFixedSize(180, 180)
        self._attendance_avatar.setAlignment(QtCore.Qt.AlignCenter)
        attendance_avatar_box_layout.addWidget(self._attendance_avatar)

        images_row = QtWidgets.QHBoxLayout()
        images_row.setContentsMargins(0, 0, 0, 0)
        images_row.addWidget(student_avatar_box)
        images_row.addWidget(attendance_avatar_box)

        self._refresh_student_avatar(student_id)
        self._refresh_attendance_avatar(attendance.id)

        # id / picker
        self._id_edit = QtWidgets.QLineEdit(str(student.id) if student else "--")
        self._id_edit.setStyleSheet("QLineEdit { background:#2a2a2a; color: gray; }")
        self._id_edit.setReadOnly(True)
        self._name_edit = QtWidgets.QLineEdit(
            student.name if student else attendance.student_name
        )

        select_student_btn = QtWidgets.QPushButton("Chọn")
        select_student_btn.setObjectName("primaryButton")
        select_student_btn.clicked.connect(self._handle_student_pick)

        # class combo
        self._class_combo = QtWidgets.QComboBox()
        self._class_combo.addItem("Chọn lớp...", None)
        for classroom in classrooms:
            self._class_combo.addItem(classroom.get("name", ""), classroom.get("_id"))
        self._class_combo.addItem("+ Thêm lớp mới", "__new__")

        self._new_class_edit = QtWidgets.QLineEdit()
        self._new_class_edit.setPlaceholderText("Nhập tên lớp mới")
        self._new_class_edit.setVisible(False)

        if student and student.class_id is not None:
            self._set_class_selection(student.class_id)

        self._class_combo.currentIndexChanged.connect(self._handle_class_change)
        self._handle_class_change()

        picker_row = QtWidgets.QHBoxLayout()
        picker_row.addWidget(self._id_edit, stretch=1)
        picker_row.addWidget(select_student_btn)

        form.addRow("", images_row)
        form.addRow("ID:", picker_row)
        form.addRow("Tên:", self._name_edit)

        class_row = QtWidgets.QVBoxLayout()
        class_row.addWidget(self._class_combo)
        class_row.addWidget(self._new_class_edit)
        form.addRow("Lớp:     ", class_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_avatar(path: str) -> QtGui.QPixmap:
        print(path)
        if not path or not os.path.exists(path):
            print("File does not exist")
            return QtGui.QPixmap()
        return QtGui.QPixmap(path).scaled(
            180, 180, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )

    def _refresh_student_avatar(self, student_id: Optional[ObjectId]) -> None:
        image_path = get_student_avatar_path(self._avatar_dir, student_id)
        if image_path and os.path.exists(image_path):
            self._student_avatar.setPixmap(self._load_avatar(image_path))
        else:
            self._student_avatar.clear()

    def _refresh_attendance_avatar(self, attendance_id: Optional[ObjectId]) -> None:
        image_path = get_attendance_frame_path(
            self._checkin_dir, self._attendance.time, attendance_id,
        )
        if image_path and os.path.exists(image_path):
            self._attendance_avatar.setPixmap(self._load_avatar(image_path))
        else:
            self._attendance_avatar.clear()

    def _set_class_selection(self, class_id) -> None:
        if class_id is None:
            return
        for idx in range(self._class_combo.count()):
            if str(self._class_combo.itemData(idx)) == str(class_id):
                self._class_combo.setCurrentIndex(idx)
                break

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------

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
        selected_student_id = (
            self._selected_student.id if self._selected_student is not None else None
        )
        new_class_id = self._class_combo.currentData()

        if not new_name:
            QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên học sinh.")
            return

        if new_class_id == "__new__":
            new_class_name = self._new_class_edit.text().strip()
            if not new_class_name:
                QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng nhập tên lớp mới.")
                return
            client = MongoClientSingleton.get_client()
            ins = client.db["classrooms"].insert_one({"name": new_class_name})
            new_class_id = ins.inserted_id
            class_name = new_class_name
        elif new_class_id is None:
            QtWidgets.QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng chọn lớp.")
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

    # ------------------------------------------------------------------
    # static convenience
    # ------------------------------------------------------------------

    @staticmethod
    def show(
        parent: QtWidgets.QWidget,
        attendance: Attendance,
        student: Optional[Student],
        student_id: Optional[ObjectId],
        classrooms: list[dict],
        avatar_dir: str,
        checkin_dir: str,
    ) -> Optional[UpdateStudentResult]:
        """Open the dialog and return the result, or ``None`` on cancel."""
        dlg = UpdateStudentDialog(
            parent, attendance, student, student_id,
            classrooms, avatar_dir, checkin_dir,
        )
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        return dlg.result
