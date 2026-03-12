import math
import numpy as np
from PySide6.QtCore import QObject, Signal

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[Tracker] scipy not found → fallback greedy match")

class KalmanBox:

    def __init__(self, cx, cy):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)

        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        self.Q = np.eye(4, dtype=float) * 0.5
        self.Q[2:, 2:] *= 10.0  

        self.R = np.eye(2, dtype=float) * 1.0

        self.P = np.eye(4, dtype=float) * 10.0

        self._predicted_this_frame = False

    def predict(self):
        if not self._predicted_this_frame:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            self._predicted_this_frame = True
        return self.x[0], self.x[1]

    def update(self, cx, cy):
        self._predicted_this_frame = False

        z = np.array([cx, cy], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def begin_frame(self):
        self._predicted_this_frame = False

    @property
    def cx(self): return self.x[0]
    @property
    def cy(self): return self.x[1]
    @property
    def vx(self): return self.x[2]
    @property
    def vy(self): return self.x[3]
    @property
    def speed(self): return math.hypot(self.x[2], self.x[3])


def iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(cost_matrix, threshold):
    n_det, n_trk = cost_matrix.shape
    matched = []
    used_det = set()
    used_trk = set()
    pairs = np.argwhere(cost_matrix < threshold)
    pairs = sorted(pairs, key=lambda p: cost_matrix[p[0], p[1]])
    for d, t in pairs:
        if d not in used_det and t not in used_trk:
            matched.append((d, t))
            used_det.add(d)
            used_trk.add(t)
    unmatched_det = [i for i in range(n_det) if i not in used_det]
    unmatched_trk = [i for i in range(n_trk) if i not in used_trk]
    return matched, unmatched_det, unmatched_trk



#  TRACKER
class Tracker(QObject):
    on_disappeared_signal = Signal(int)
    def __init__(
        self,
        parent=None,
        max_disappeared=3,  
        base_dist_thresh=400,
        iou_weight=0.4,
        dist_weight=0.6,
        min_iou_thresh=0.05,
        fast_speed_thresh=15.0,
        match_cost_thresh=1.5,
    ):
        super().__init__(parent)
        self.max_disappeared = max_disappeared
        self.base_dist_thresh = base_dist_thresh
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight
        self.min_iou_thresh = min_iou_thresh
        self.fast_speed_thresh = fast_speed_thresh
        self.match_cost_thresh = match_cost_thresh

        self.kalman: dict[int, KalmanBox] = {}
        self.id_to_bbox: dict[int, tuple] = {}
        self.id_to_pred_bbox: dict[int, tuple] = {}
        self.id_to_class: dict[int, str] = {}
        self.disappeared: dict[int, int] = {}

        self.id_count = 0
        self.objects_bbs_ids = []

        self.counting: dict[str, int] = {}
        self.countingTotal: dict[str, int] = {}

    # ──────────────────────────────────────────
    def _adaptive_dist_thresh(self, obj_id, w, h):

        obj_size = (w + h) / 2
        base = max(self.base_dist_thresh, obj_size * 0.7)

        speed = self.kalman[obj_id].speed if obj_id in self.kalman else 0
        # [FIX] Tăng hệ số speed từ 2.5 → 4.0, cap từ 200 → 400
        speed_bonus = min(speed * 4.5, 500)

        return base + speed_bonus

    # ──────────────────────────────────────────
    def _build_cost_matrix(self, detections, track_ids):
        n_det = len(detections)
        n_trk = len(track_ids)
        cost = np.full((n_det, n_trk), fill_value=1e6, dtype=float)

        for di, (x1, y1, x2, y2) in enumerate(detections):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            for ti, tid in enumerate(track_ids):
                kf = self.kalman[tid]
                px, py = kf.cx, kf.cy

                dist = math.hypot(cx - px, cy - py)
                thresh = self._adaptive_dist_thresh(tid, w, h)

                if dist > thresh:
                    continue

                norm_dist = dist / thresh

                pbbox = self.id_to_pred_bbox.get(tid)
                iou_score = iou((x1, y1, x2, y2), pbbox) if pbbox else 0.0

                is_fast = kf.speed > self.fast_speed_thresh

                if is_fast:
                    cost[di, ti] = norm_dist
                else:
                    if iou_score < self.min_iou_thresh and pbbox is not None:
                        iou_penalty = 0.3   
                    else:
                        iou_penalty = 0.0

                    cost[di, ti] = (
                        self.dist_weight  * norm_dist
                        + self.iou_weight * (1.0 - iou_score)
                        + iou_penalty
                    )

        return cost

    # ──────────────────────────────────────────
    def predict(self):
        result = []
        for tid, kf in self.kalman.items():
            if tid not in self.id_to_bbox:
                continue
            kf.begin_frame()  
            px, py = kf.predict()
            rx1, ry1, rx2, ry2 = self.id_to_bbox[tid]
            w = rx2 - rx1
            h = ry2 - ry1
            nx1 = int(px - w / 2)
            ny1 = int(py - h / 2)
            nx2 = nx1 + w
            ny2 = ny1 + h
            self.id_to_pred_bbox[tid] = (nx1, ny1, nx2, ny2)
            classId = self.id_to_class.get(tid, "face")
            result.append([nx1, ny1, nx2, ny2, tid, classId])
        return result

    # ──────────────────────────────────────────
    def update(self, objects_rect, classId="face"):
        if classId not in self.counting:
            self.counting[classId] = 0
        if classId not in self.countingTotal:
            self.countingTotal[classId] = 0

        for tid, kf in self.kalman.items():
            kf.begin_frame()
            if self.disappeared.get(tid, 0) == 0:
                px, py = kf.predict()
            else:
                px, py = kf.cx, kf.cy
            if tid in self.id_to_bbox:
                rx1, ry1, rx2, ry2 = self.id_to_bbox[tid]
                w = rx2 - rx1
                h = ry2 - ry1
                nx1 = int(px - w / 2)
                ny1 = int(py - h / 2)
                self.id_to_pred_bbox[tid] = (nx1, ny1, nx1 + w, ny1 + h)

        track_ids = [
            tid for tid in self.kalman
            if self.id_to_class.get(tid) == classId
            and self.disappeared.get(tid, 0) == 0 
        ]

        objects_now = []
        matched_det = set()
        matched_trk = set()

        if objects_rect and track_ids:
            cost = self._build_cost_matrix(objects_rect, track_ids)

            if HAS_SCIPY:
                row_ind, col_ind = linear_sum_assignment(cost)
                pairs = [(r, c) for r, c in zip(row_ind, col_ind)
                         if cost[r, c] < self.match_cost_thresh]
            else:
                pairs, _, _ = greedy_match(cost, threshold=self.match_cost_thresh)

            for di, ti in pairs:
                x1, y1, x2, y2 = objects_rect[di]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tid = track_ids[ti]

                self.kalman[tid].update(cx, cy)
                self.id_to_bbox[tid] = (x1, y1, x2, y2)
                self.id_to_pred_bbox[tid] = (x1, y1, x2, y2)
                self.disappeared.pop(tid, None)

                objects_now.append([x1, y1, x2, y2, tid, classId])
                matched_det.add(di)
                matched_trk.add(ti)

        disappeared_ids = [
            tid for tid in self.disappeared
            if self.id_to_class.get(tid) == classId
            and tid in self.kalman
            and tid in self.id_to_bbox
        ]

        reidentified_det = set()
        if disappeared_ids:
            unmatched_dets = [i for i in range(len(objects_rect)) if i not in matched_det]
            for di in unmatched_dets:
                x1, y1, x2, y2 = objects_rect[di]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1

                best_tid  = None
                best_dist = float("inf")
                for tid in disappeared_ids:
                    lx1, ly1, lx2, ly2 = self.id_to_bbox[tid]
                    last_cx = (lx1 + lx2) / 2
                    last_cy = (ly1 + ly2) / 2
                    dist = math.hypot(cx - last_cx, cy - last_cy)
                    thresh = self._adaptive_dist_thresh(tid, w, h)
                    if dist < thresh and dist < best_dist:
                        best_dist = dist
                        best_tid  = tid

                if best_tid is not None:
                    self.kalman[best_tid].update(cx, cy)
                    self.id_to_bbox[best_tid]      = (x1, y1, x2, y2)
                    self.id_to_pred_bbox[best_tid] = (x1, y1, x2, y2)
                    prev_missing = self.disappeared.pop(best_tid, 0)
                    objects_now.append([x1, y1, x2, y2, best_tid, classId])
                    matched_det.add(di)
                    reidentified_det.add(di)
                    disappeared_ids.remove(best_tid)

        for di, rect in enumerate(objects_rect):
            if di in matched_det or di in reidentified_det:
                continue
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            new_id = self.id_count
            self.id_count += 1

            self.kalman[new_id] = KalmanBox(cx, cy)
            self.kalman[new_id].update(cx, cy)
            self.id_to_bbox[new_id] = (x1, y1, x2, y2)
            self.id_to_pred_bbox[new_id] = (x1, y1, x2, y2)
            self.id_to_class[new_id] = classId
            self.disappeared[new_id] = 0

            objects_now.append([x1, y1, x2, y2, new_id, classId])
            self.countingTotal[classId] += 1

        active_in_result = {int(obj[4]) for obj in objects_now}
        for tid in list(self.kalman.keys()):
            if self.id_to_class.get(tid) != classId:
                continue
            if tid not in active_in_result:
                self.disappeared[tid] = self.disappeared.get(tid, 0) + 1

        to_remove = [
            tid for tid, cnt in self.disappeared.items()
            if cnt > self.max_disappeared
        ]
        for tid in to_remove:
            self.on_disappeared_signal.emit(tid)
            self.disappeared.pop(tid, None)
            self.kalman.pop(tid, None)
            self.id_to_bbox.pop(tid, None)
            self.id_to_pred_bbox.pop(tid, None)
            self.id_to_class.pop(tid, None)

        self.counting[classId] = len(objects_now)
        self.objects_bbs_ids = [
            obj for obj in self.objects_bbs_ids if obj[5] != classId
        ]
        self.objects_bbs_ids.extend(objects_now)

        return objects_now
    
    def release(self):
        for obj in self.objects_bbs_ids:
            obj_id = obj[4]
            self.on_disappeared_signal.emit(obj_id)
            self.disappeared.pop(obj_id, None)
            self.kalman.pop(obj_id, None)
            self.id_to_bbox.pop(obj_id, None)
            self.id_to_pred_bbox.pop(obj_id, None)
            self.id_to_class.pop(obj_id, None)
    # ──────────────────────────────────────────
    @property
    def center_points(self):
        return {tid: (kf.cx, kf.cy) for tid, kf in self.kalman.items()}

    @property
    def velocities(self):
        return {tid: (kf.vx, kf.vy) for tid, kf in self.kalman.items()}
