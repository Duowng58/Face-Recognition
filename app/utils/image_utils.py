import os
from typing import Optional

from bson import ObjectId

from app.utils.mongodb_access import to_local


def get_attendance_frame_path(checkin_dir: str, record_time, attendance_id: Optional[ObjectId]) -> Optional[str]:
    if attendance_id is None:
        return None
    record_date = to_local(record_time).strftime("%Y-%m-%d")
    return os.path.join(checkin_dir, record_date, str(attendance_id), "frame.jpg")


def get_student_avatar_path(avatar_dir: str, student_id: Optional[ObjectId]) -> Optional[str]:
    if student_id is None:
        return None
    return os.path.join(avatar_dir, f"{student_id}.jpg")
