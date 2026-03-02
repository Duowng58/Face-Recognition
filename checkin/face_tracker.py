import math

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.velocities = {}
        self.last_positions = {}
        self.id_to_class = {}
        self.objects_bbs_ids = []

        # object đã mất tạm thời
        self.disappeared_objects = {}
        self.max_disappeared_frames = 600

        self.id_count = 0
        self.counting = {}
        self.countingTotal = {}

    # ------------------------------------------------
    def calculate_distance_threshold(self, w, h):
        obj_size = (w + h) / 2
        return max(35, min(obj_size * 0.5, 120))

    # ------------------------------------------------
    def predict_next_position(self, obj_id):
        if obj_id in self.center_points and obj_id in self.velocities:
            x, y = self.center_points[obj_id]
            vx, vy = self.velocities[obj_id]
            return (x + vx, y + vy)
        return None

    # ------------------------------------------------
    def update(self, objects_rect, classId="face"):
        """
        objects_rect: list of [x1, y1, x2, y2]
        return: [x1, y1, x2, y2, id, classId]
        """

        objects_now = []
        used_ids = set()

        # init statistics key
        if classId not in self.counting:
            self.counting[classId] = 0
        if classId not in self.countingTotal:
            self.countingTotal[classId] = 0

        # ================= MATCH OBJECT =================
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            best_match_id = None
            min_distance = float("inf")

            threshold = self.calculate_distance_threshold(
                x2 - x1, y2 - y1
            )

            # ---- match with active objects ----
            for obj_id, center in self.center_points.items():
                if obj_id in used_ids:
                    continue

                if self.id_to_class.get(obj_id) != classId:
                    continue

                predicted = self.predict_next_position(obj_id)
                if predicted:
                    px, py = predicted
                    dist = math.hypot(cx - px, cy - py)
                else:
                    dist = math.hypot(cx - center[0], cy - center[1])

                if dist < threshold and dist < min_distance:
                    min_distance = dist
                    best_match_id = obj_id

            # ---- match with disappeared objects ----
            if best_match_id is None:
                for obj_id, info in self.disappeared_objects.items():
                    if info["class"] != classId or obj_id in used_ids:
                        continue

                    px, py = info["center"]
                    dist = math.hypot(cx - px, cy - py)

                    if dist < threshold * 1.5 and dist < min_distance:
                        min_distance = dist
                        best_match_id = obj_id

            # ================= UPDATE =================
            if best_match_id is not None:
                # remove from disappeared
                self.disappeared_objects.pop(best_match_id, None)

                old_x, old_y = self.center_points.get(best_match_id, (cx, cy))
                self.velocities[best_match_id] = (cx - old_x, cy - old_y)

                self.last_positions[best_match_id] = (old_x, old_y)
                self.center_points[best_match_id] = (cx, cy)

                objects_now.append([x1, y1, x2, y2, best_match_id, classId])
                used_ids.add(best_match_id)

            else:
                # new object
                new_id = self.id_count
                self.id_count += 1

                self.center_points[new_id] = (cx, cy)
                self.velocities[new_id] = (0, 0)
                self.last_positions[new_id] = (cx, cy)
                self.id_to_class[new_id] = classId

                objects_now.append([x1, y1, x2, y2, new_id, classId])
                used_ids.add(new_id)

                self.countingTotal[classId] += 1

        # ================= HANDLE DISAPPEARED =================
        active_ids = {obj[4] for obj in objects_now}

        for obj in self.objects_bbs_ids:
            obj_id = obj[4]
            if obj_id not in active_ids and obj_id not in self.disappeared_objects:
                self.disappeared_objects[obj_id] = {
                    "center": self.center_points.get(obj_id, (0, 0)),
                    "class": classId,
                    "frames_missing": 1,
                    "velocity": self.velocities.get(obj_id, (0, 0))
                }

        # increase missing counter
        to_remove = []
        for obj_id, info in self.disappeared_objects.items():
            info["frames_missing"] += 1
            if info["frames_missing"] > self.max_disappeared_frames:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            self.disappeared_objects.pop(obj_id, None)
            self.center_points.pop(obj_id, None)
            self.velocities.pop(obj_id, None)
            self.last_positions.pop(obj_id, None)
            self.id_to_class.pop(obj_id, None)

        # ================= CLEAN & UPDATE =================
        self.objects_bbs_ids = [
            obj for obj in self.objects_bbs_ids if obj[5] != classId
        ]
        self.objects_bbs_ids.extend(objects_now)

        self.counting[classId] = len(objects_now)

        # keep only active + disappeared
        keep_ids = set(active_ids).union(self.disappeared_objects.keys())

        self.center_points = {
            i: v for i, v in self.center_points.items() if i in keep_ids
        }
        self.velocities = {
            i: v for i, v in self.velocities.items() if i in keep_ids
        }
        self.last_positions = {
            i: v for i, v in self.last_positions.items() if i in keep_ids
        }
        self.id_to_class = {
            i: v for i, v in self.id_to_class.items() if i in keep_ids
        }

        return objects_now