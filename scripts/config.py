"""
Shared configuration constants.

This module is framework-agnostic and can be imported by any UI
(PySide6, Tkinter, headless, etc.).
"""

import os

import cv2

# ── paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

FACE_DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, "face_data_1"))
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH = os.path.join(FACE_DATA_DIR, "image_paths.json")

FACE_DATA_UNKNOWN_DIR = os.path.normpath(os.path.join(ROOT_DIR, "face_data_unknown"))
ANNOY_UNKNOWN_INDEX_PATH = os.path.join(FACE_DATA_UNKNOWN_DIR, "face_index_unknown.ann")
MAPPING_UNKNOWN_PATH = os.path.join(FACE_DATA_UNKNOWN_DIR, "image_paths_unknown.json")

# ── recognition parameters ────────────────────────────────────
MIN_BBOX_AREA = 1024
EMBEDDING_DIM = 512
TREE = 50
SIM_THRESHOLD = 0.6

# ── capture / storage ────────────────────────────────────────
CAPTURE_ROOT = "captured_faces"
KNOWN_DIR = os.path.join(CAPTURE_ROOT, "known")
CHECKIN_DIR = os.path.join(CAPTURE_ROOT, "checkin")
UNKNOWN_DIR = os.path.join(CAPTURE_ROOT, "unknown")
AVATAR_DIR = os.path.join(CAPTURE_ROOT, "avatars")

VIDEO_FPS = 15
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# ── network ───────────────────────────────────────────────────
DEFAULT_RTSP_URL = "rtsp://admin:Ancovn12@192.168.1.231:554/Streaming/Channels/201/video"
DEFAULT_RTMP_URL = "rtmp://124.158.7.217:5001/LiveStream/detect_01"
VIDEO = os.path.normpath(os.path.join(ROOT_DIR, "assets/videos/record-2026-03-24-06-56-53.mkv"))

# ── font ──────────────────────────────────────────────────────
FONT_PATH = os.path.join(ROOT_DIR, "assets", "fonts", "Arial.ttf")

# ── frame dimensions ──────────────────────────────────────────
FRAME_WIDTH = 1344
FRAME_HEIGHT = 768
