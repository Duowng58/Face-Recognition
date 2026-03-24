"""
Qt-compatible Tracker that keeps QObject + Signal interface for PySide6 UI,
while delegating all core logic to scripts.services.face_tracker.

For non-Qt UIs (Tkinter, etc.), import directly from scripts.services.face_tracker.
"""

from PySide6.QtCore import QObject, Signal

# Re-export helpers so existing imports keep working
from scripts.services.face_tracker import KalmanBox, iou, greedy_match  # noqa: F401
from scripts.services.face_tracker import Tracker as _BaseTracker


class Tracker(QObject):
    """Tracker subclass that emits a Qt Signal when a track disappears."""

    on_disappeared_signal = Signal(int)

    def __init__(
        self,
        parent=None,
        max_disappeared=20,
        base_dist_thresh=400,
        iou_weight=0.5,
        dist_weight=0.5,
        min_iou_thresh=0.1,
        fast_speed_thresh=15.0,
        match_cost_thresh=1.5,
    ):
        super().__init__(parent)

        # Delegate to the framework-agnostic tracker
        self._base = _BaseTracker(
            max_disappeared=max_disappeared,
            base_dist_thresh=base_dist_thresh,
            iou_weight=iou_weight,
            dist_weight=dist_weight,
            min_iou_thresh=min_iou_thresh,
            fast_speed_thresh=fast_speed_thresh,
            match_cost_thresh=match_cost_thresh,
        )
        # Wire callback → Qt Signal
        self._base.on_disappeared = lambda tid: self.on_disappeared_signal.emit(tid)

    # ── Forward every public attribute / method to _base ──

    @property
    def kalman(self):
        return self._base.kalman

    @property
    def id_to_bbox(self):
        return self._base.id_to_bbox

    @id_to_bbox.setter
    def id_to_bbox(self, v):
        self._base.id_to_bbox = v

    @property
    def id_to_pred_bbox(self):
        return self._base.id_to_pred_bbox

    @property
    def id_to_class(self):
        return self._base.id_to_class

    @property
    def disappeared(self):
        return self._base.disappeared

    @property
    def id_count(self):
        return self._base.id_count

    @id_count.setter
    def id_count(self, v):
        self._base.id_count = v

    @property
    def objects_bbs_ids(self):
        return self._base.objects_bbs_ids

    @objects_bbs_ids.setter
    def objects_bbs_ids(self, v):
        self._base.objects_bbs_ids = v

    @property
    def counting(self):
        return self._base.counting

    @property
    def countingTotal(self):
        return self._base.countingTotal

    @property
    def center_points(self):
        return self._base.center_points

    @property
    def velocities(self):
        return self._base.velocities

    def predict(self):
        return self._base.predict()

    def update(self, objects_rect, classId="face"):
        return self._base.update(objects_rect, classId)

    def release(self):
        return self._base.release()
