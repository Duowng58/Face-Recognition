"""
MongoDB access layer – framework-agnostic.

Data classes (Student, Attendance), repository helpers, and timezone utilities.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable, Optional

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# ── timezone helpers ──────────────────────────────────────────

LOCAL_TZ = timezone(timedelta(hours=7))


def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def start_of_today_local() -> datetime:
    now = now_local()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def to_local(dt_value: Any) -> datetime:
    if dt_value is None:
        return now_local()
    if isinstance(dt_value, str):
        try:
            dt_value = datetime.fromisoformat(dt_value)
        except ValueError:
            return now_local()
    if not isinstance(dt_value, datetime):
        return now_local()
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    return dt_value.astimezone(LOCAL_TZ)


# ── config ────────────────────────────────────────────────────


@dataclass
class MongoConfig:
    uri: str
    database: str


default_config = MongoConfig(
    uri=os.getenv("MONGODB_URI", "mongodb://admin:admin@localhost:27017"),
    database=os.getenv("MONGODB_DB", "student_attendance"),
)


# ── singleton client ──────────────────────────────────────────


class MongoClientSingleton:
    """Singleton MongoClient with auto-reconnect on ping failure."""

    _lock = threading.Lock()
    _client: Optional["MongoDBClient"] = None
    _config: Optional[MongoConfig] = None

    @classmethod
    def configure(cls, config: MongoConfig) -> None:
        cls._config = config

    @classmethod
    def get_client(cls, config: Optional[MongoConfig] = None) -> "MongoDBClient":
        if config is not None:
            cls._config = config
        if cls._config is None:
            cls._config = default_config

        with cls._lock:
            if cls._client is None:
                cls._client = MongoDBClient(cls._config)
                cls._client.connect()
                return cls._client
            try:
                cls._client.db.command("ping")
            except Exception:
                cls._client.close()
                cls._client = MongoDBClient(cls._config)
                cls._client.connect()
            return cls._client

    @classmethod
    def get_database(cls, name: Optional[str] = None) -> Database:
        client = cls.get_client()
        db_name = name or (cls._config.database if cls._config else "student_attendance")
        return client[db_name]

    @classmethod
    def close(cls) -> None:
        with cls._lock:
            if cls._client is not None:
                cls._client.close()
            cls._client = None


# ── low-level client ──────────────────────────────────────────


class MongoDBClient:
    """MongoDB helper for CRUD operations."""

    def __init__(self, config: Optional[MongoConfig] = None) -> None:
        if config is None:
            config = default_config
        self._config = config
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None

    @property
    def db(self) -> Database:
        if self._db is None:
            raise RuntimeError("MongoDB is not connected. Call connect() first.")
        return self._db

    def connect(self) -> None:
        self._client = MongoClient(self._config.uri)
        self._db = self._client[self._config.database]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None

    def collection(self, name: str) -> Collection:
        return self.db[name]

    def find(self, collection: str, query: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        query = query or {}
        return list(self.collection(collection).find(query))

    def find_one(self, collection: str, query: dict[str, Any]) -> Optional[dict[str, Any]]:
        return self.collection(collection).find_one(query)

    def insert_one(self, collection: str, data: dict[str, Any]) -> str:
        result = self.collection(collection).insert_one(data)
        return str(result.inserted_id)

    def insert_many(self, collection: str, data: Iterable[dict[str, Any]]) -> list[str]:
        result = self.collection(collection).insert_many(list(data))
        return [str(item_id) for item_id in result.inserted_ids]

    def update_one(
        self,
        collection: str,
        query: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> int:
        result = self.collection(collection).update_one(query, {"$set": update}, upsert=upsert)
        return result.modified_count

    def delete_one(self, collection: str, query: dict[str, Any]) -> int:
        result = self.collection(collection).delete_one(query)
        return result.deleted_count

    def delete_many(self, collection: str, query: dict[str, Any]) -> int:
        result = self.collection(collection).delete_many(query)
        return result.deleted_count


# ── data models ───────────────────────────────────────────────


@dataclass
class Student:
    id: Optional[ObjectId]
    name: str
    class_id: str

    def to_document(self) -> dict[str, Any]:
        payload = {"name": self.name, "class_id": self.class_id}
        if self.id is not None:
            payload["_id"] = self.id
        return payload

    @staticmethod
    def from_document(doc: dict[str, Any]) -> "Student":
        return Student(
            id=doc.get("_id"),
            name=doc.get("name", ""),
            class_id=doc.get("class_id", ""),
        )


@dataclass
class Attendance:
    id: Optional[ObjectId] = None
    student_id: ObjectId | None = None
    student_name: str = ""
    student_classroom: str = ""
    time: datetime = field(default_factory=lambda: now_local())
    score: float = 0.0

    def to_document(self) -> dict[str, Any]:
        payload = {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "student_classroom": self.student_classroom,
            "time": self.time,
            "score": self.score,
        }
        if self.id is not None:
            payload["_id"] = self.id
        return payload

    @staticmethod
    def from_document(doc: dict[str, Any]) -> "Attendance":
        time = to_local(doc.get("time", now_local()))
        return Attendance(
            id=doc.get("_id"),
            student_id=doc.get("student_id"),
            student_name=doc.get("student_name", ""),
            student_classroom=doc.get("student_classroom", ""),
            time=time,
            score=doc.get("score", 0.0),
        )


# ── repositories ──────────────────────────────────────────────


class StudentRepository:
    """CRUD helper for the students collection."""

    def __init__(self, collection: str = "students") -> None:
        self._client = MongoClientSingleton.get_client()
        self._collection = collection

    def get(self, student_id: ObjectId) -> Optional[Student]:
        doc = self._client.find_one(self._collection, {"_id": student_id})
        return Student.from_document(doc) if doc else None

    def find(self, query: Optional[dict[str, Any]] = None) -> list[Student]:
        return [Student.from_document(doc) for doc in self._client.find(self._collection, query)]

    def insert(self, student: Student) -> ObjectId:
        return self._client.collection(self._collection).insert_one(student.to_document()).inserted_id

    def update(self, student_id: ObjectId, update: dict[str, Any]) -> int:
        return self._client.update_one(self._collection, {"_id": student_id}, update)

    def delete(self, student_id: ObjectId) -> int:
        return self._client.delete_one(self._collection, {"_id": student_id})


class AttendanceRepository:
    """CRUD helper for the attendances collection."""

    def __init__(self, collection: str = "attendances") -> None:
        self._client = MongoClientSingleton.get_client()
        self._collection = collection

    def get(self, attendance_id: ObjectId) -> Optional[Attendance]:
        doc = self._client.find_one(self._collection, {"_id": attendance_id})
        return Attendance.from_document(doc) if doc else None

    def find(self, query: Optional[dict[str, Any]] = None) -> list[Attendance]:
        return [Attendance.from_document(doc) for doc in self._client.find(self._collection, query)]

    def insert(self, attendance: Attendance) -> ObjectId:
        return self._client.collection(self._collection).insert_one(attendance.to_document()).inserted_id

    def update(self, attendance_id: ObjectId, update: dict[str, Any]) -> int:
        return self._client.update_one(self._collection, {"_id": attendance_id}, update)

    def delete(self, attendance_id: ObjectId) -> int:
        return self._client.delete_one(self._collection, {"_id": attendance_id})
