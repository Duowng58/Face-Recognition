"""
Re-export mongodb_access from scripts.utils.mongodb_access.

Existing code using ``from app.utils.mongodb_access import ...`` will continue to work.
"""

from scripts.utils.mongodb_access import (  # noqa: F401
    MongoConfig,
    default_config,
    MongoClientSingleton,
    MongoDBClient,
    Student,
    Attendance,
    LOCAL_TZ,
    now_local,
    start_of_today_local,
    to_local,
    StudentRepository,
    AttendanceRepository,
)
