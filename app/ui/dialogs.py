from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from app.utils.image_utils import get_student_avatar_path
from app.utils.mongodb_access import Student


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
