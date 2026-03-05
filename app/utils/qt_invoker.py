from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtCore import QObject, Signal, QTimer


class _QtInvoker(QObject):
    """
    Bộ đệm gọi hàm trên main thread của Qt bằng signal/slot.
    Dùng để dispatch callback từ các thread phụ (threading.Thread) về UI thread an toàn.
    """

    invoke_signal = Signal(object, int)  # (callable, delay_ms)

    def __init__(self) -> None:
        super().__init__()
        self.invoke_signal.connect(self._on_invoke)

    def _on_invoke(self, cb: Callable[[], None], delay_ms: int) -> None:
        if delay_ms and delay_ms > 0:
            QTimer.singleShot(delay_ms, cb)
        else:
            cb()


_invoker_singleton: Optional[_QtInvoker] = None


def init_qt_invoker() -> _QtInvoker:
    """
    Khởi tạo singleton invoker trên main thread (gọi trong main sau khi tạo QApplication).
    """
    global _invoker_singleton
    if _invoker_singleton is None:
        _invoker_singleton = _QtInvoker()
    return _invoker_singleton


def qt_invoke(cb: Callable[[], None], delay_ms: int = 0) -> None:
    """Emit signal để chạy cb trên main thread với delay tuỳ chọn."""
    if _invoker_singleton is not None:
        _invoker_singleton.invoke_signal.emit(cb, delay_ms)
    # else:
    #     # Chưa init: gọi trực tiếp (fallback)
    #     cb()
