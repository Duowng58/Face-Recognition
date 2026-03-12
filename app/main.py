import os
import sys

from PySide6 import QtWidgets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.ui.attendance_window import AttendanceWindow
from app.utils.qt_invoker import init_qt_invoker


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    init_qt_invoker()
    window = AttendanceWindow()
    app.aboutToQuit.connect(window._stop_capture)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
