import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

FACE_DATA_DIR = os.path.normpath(os.path.join(ROOT_DIR, "face_data_1"))
ANNOY_INDEX_PATH = os.path.join(FACE_DATA_DIR, "face_index.ann")
MAPPING_PATH = os.path.join(FACE_DATA_DIR, "image_paths.json")

MIN_BBOX_AREA = 10000
EMBEDDING_DIM = 512
TREE = 50
SIM_THRESHOLD = 0.6

CAPTURE_ROOT = "captured_faces"
KNOWN_DIR = os.path.join(CAPTURE_ROOT, "known")
CHECKIN_DIR = os.path.join(CAPTURE_ROOT, "checkin")
UNKNOWN_DIR = os.path.join(CAPTURE_ROOT, "unknown")
AVATAR_DIR = os.path.join(CAPTURE_ROOT, "avatars")

VIDEO_FPS = 15
VIDEO_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

DEFAULT_RTSP_URL = "rtsp://admin:Ancovn1234@192.168.1.64:554/Streaming/Channels/201/video"
DEFAULT_RTMP_URL = "rtmp://124.158.7.217:5001/LiveStream/detect_01"

FONT_PATH = os.path.join(ROOT_DIR, "app", "fonts", "Arial.ttf")

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
