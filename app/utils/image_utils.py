"""
Re-export image utils from scripts.utils.image_utils.

Existing code using ``from app.utils.image_utils import ...`` will continue to work.
"""

from scripts.utils.image_utils import get_attendance_frame_path, get_student_avatar_path  # noqa: F401
