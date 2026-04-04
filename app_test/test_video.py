"""
Test video face recognition using a pre-built Annoy index.

Usage:
    python app_test/test_video.py                                  # default video from config
    python app_test/test_video.py --video "C:/path/to/video.mp4"   # custom video
    python app_test/test_video.py --source rtsp                    # RTSP stream
    python app_test/test_video.py --source 0                       # webcam
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import datetime

import cv2
import numpy as np
from annoy import AnnoyIndex
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ── Ensure project root is on sys.path ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.config import (
    ANNOY_INDEX_PATH,
    DEFAULT_RTSP_URL,
    DETECT_SIZE,
    EMBEDDING_DIM,
    FACE_DATA_DIR,
    FONT_PATH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAPPING_PATH,
    SIM_THRESHOLD,
    VIDEO,
)
from scripts.utils.cv2_helper import cv2_putText_utf8


# ─────────────────────────────────────────────────────────────
# Face-analysis model (lazy singleton)
# ─────────────────────────────────────────────────────────────

_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is not None:
        return _face_app

    trt_options = {
        "device_id": 0,
        "trt_max_workspace_size": 1 << 30,
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": os.path.abspath("./trt_cache"),
    }
    _face_app = FaceAnalysis(
        name="buffalo_s",
        root="./my_models",
        providers=[
            ("TensorrtExecutionProvider", trt_options),
            "CUDAExecutionProvider",
        ],
        allowed_modules=["detection", "recognition"],
    )
    _face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=DETECT_SIZE)

    # Warm-up
    dummy = np.zeros((DETECT_SIZE[1], DETECT_SIZE[0], 3), dtype=np.uint8)
    _face_app.get(dummy)
    print("[OK] Face model loaded & warmed up.")
    return _face_app


# ─────────────────────────────────────────────────────────────
# Annoy index
# ─────────────────────────────────────────────────────────────

def load_annoy_index() -> tuple[AnnoyIndex | None, dict[str, str]]:
    if not os.path.exists(ANNOY_INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print(f"[WARN] Index not found: {ANNOY_INDEX_PATH}")
        return None, {}

    ann = AnnoyIndex(EMBEDDING_DIM, "angular")
    ann.load(ANNOY_INDEX_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        idx2name = json.load(f)

    unique = set(idx2name.values())
    print(f"[OK] Annoy index loaded: {ann.get_n_items()} vectors, {len(unique)} people")
    return ann, idx2name


def recognize(ann: AnnoyIndex, idx2name: dict, embedding: np.ndarray) -> tuple[str, float]:
    """Return (name, similarity). 'Unknown' if below threshold."""
    if ann is None or ann.get_n_items() == 0:
        return "Unknown", 0.0

    idx_list, dist_list = ann.get_nns_by_vector(embedding, 1, include_distances=True)
    if not dist_list:
        return "Unknown", 0.0

    sim = 1 - (dist_list[0] ** 2) / 2
    name = idx2name.get(str(idx_list[0]), "Unknown")
    print(f"  [DEBUG] idx_list: {idx_list}, dist_list: {dist_list}, sim: {sim}, name: {name}")
    
    if sim >= SIM_THRESHOLD:
        return name, sim
    return "Unknown", sim


# ─────────────────────────────────────────────────────────────
# Letterbox helpers
# ─────────────────────────────────────────────────────────────

def _letterbox_params(orig_h: int, orig_w: int):
    tw, th = DETECT_SIZE
    scale = min(tw / orig_w, th / orig_h)
    offset_x = (tw - int(orig_w * scale)) // 2
    offset_y = (th - int(orig_h * scale)) // 2
    return scale, offset_x, offset_y


def _letterbox(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale, ox, oy = _letterbox_params(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((DETECT_SIZE[1], DETECT_SIZE[0], 3), 128, dtype=np.uint8)
    canvas[oy: oy + resized.shape[0], ox: ox + resized.shape[1]] = resized
    return canvas


def _map_to_original(coords, scale, ox, oy):
    return (np.array(coords) - np.array([ox, oy])) / scale


# ─────────────────────────────────────────────────────────────
# Detect + Recognize on a single frame
# ─────────────────────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    ann: AnnoyIndex | None,
    idx2name: dict,
) -> list[dict]:
    """
    Detect faces, extract embeddings, recognize.
    Returns list of dicts: {bbox, name, sim, det_score}.
    """
    app = get_face_app()
    h, w = frame.shape[:2]
    
    
    
    
    
    

    results: list[dict] = []
    
    
    # faces = app.get(frame)
    # for face in faces:
    #     normed = face.normed_embedding
    #     name, sim = recognize(ann, idx2name, normed)
    #     results.append({
    #         "bbox": face.bbox,
    #         "name": name,
    #         "sim": sim,
    #         "det_score": face.det_score,
    #     })
    # return results 
        
        
    small = _letterbox(frame)
    scale, ox, oy = _letterbox_params(h, w)

    bboxes, kpss = app.models["detection"].detect(small)
    if bboxes.shape[0] == 0:
        return results

    for i in range(bboxes.shape[0]):
        try:
            x1_sm, y1_sm, x2_sm, y2_sm, det_score = bboxes[i]
            x1 = int((x1_sm - ox) / scale)
            y1 = int((y1_sm - oy) / scale)
            x2 = int((x2_sm - ox) / scale)
            y2 = int((y2_sm - oy) / scale)

            kps_orig = _map_to_original(kpss[i], scale, ox, oy)
            face_aimg = face_align.norm_crop(frame, kps_orig)
            feat = app.models["recognition"].get_feat(face_aimg)
            normed = (feat / np.linalg.norm(feat)).flatten()

            name, sim = recognize(ann, idx2name, normed)

            results.append({
                "bbox": (x1, y1, x2, y2),
                "name": name,
                "sim": sim,
                "det_score": det_score,
            })
        except Exception as e:
            print(f"  [WARN] Error processing face {i}: {e}")

    return results


# ─────────────────────────────────────────────────────────────
# Draw results on frame
# ─────────────────────────────────────────────────────────────

def draw_results(
    frame: np.ndarray,
    faces: list[dict],
    stats: list[str] | None = None,
) -> np.ndarray:
    """Draw bounding boxes, names, and optional stats on frame."""
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        name = face["name"]
        sim = face["sim"]
        det = face["det_score"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({sim:.2f})" if name != "Unknown" else f"Unknown ({sim:.2f})"
        label += f" det:{det:.2f}"

        if os.path.exists(FONT_PATH):
            frame = cv2_putText_utf8(frame, label, (x1, y1 - 35), FONT_PATH, 28, color)
        else:
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if stats:
        for i, text in enumerate(stats):
            y = 30 + i * 30
            cv2.putText(frame, text, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return frame


# ─────────────────────────────────────────────────────────────
# Parse video start time from filename
# ─────────────────────────────────────────────────────────────

def parse_start_time(filepath: str) -> datetime.datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", filepath)
    if match:
        return datetime.datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
    return datetime.datetime.now()


# ─────────────────────────────────────────────────────────────
# Resolve source
# ─────────────────────────────────────────────────────────────

def resolve_source(raw: str):
    low = raw.strip().lower()
    if low in ("0", "webcam"):
        return 0
    if low in ("rtsp", "rtsp_default"):
        return DEFAULT_RTSP_URL
    return raw.strip()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test video face recognition với Annoy index đã build.",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Nguồn video: 0/webcam, rtsp, hoặc đường dẫn file. Mặc định: video trong config.",
    )
    args = parser.parse_args()

    # ── Resolve source ────────────────────────────────────────
    if args.source:
        source = resolve_source(args.source)
    else:
        source = VIDEO

    print("=" * 60)
    print(f"  Source: {source}")
    print(f"  Index:  {ANNOY_INDEX_PATH}")
    print(f"  Threshold: {SIM_THRESHOLD}")
    print("=" * 60)

    # ── Load model + index ────────────────────────────────────
    get_face_app()
    ann, idx2name = load_annoy_index()

    # ── Open video ────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được nguồn: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = parse_start_time(str(source)) if isinstance(source, str) else datetime.datetime.now()

    print(f"  FPS: {fps:.1f}  |  Total frames: {total_frames}")
    print(f"  Start time: {start_time}")
    print(f"  Press 'q' to quit.\n")

    frame_count = 0
    target_duration = 1.0 / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        frame_count += 1
        t0 = time.perf_counter()

        # ── Detect + Recognize ────────────────────────────────
        faces = process_frame(frame, ann, idx2name)

        detect_ms = (time.perf_counter() - t0) * 1000

        # ── Compute current video time ────────────────────────
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time = start_time + datetime.timedelta(milliseconds=msec)
        time_display = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # ── Draw ──────────────────────────────────────────────
        stats = [
            f"Frame: {frame_count}/{total_frames}  |  FPS: {fps:.0f}",
            f"Detect: {detect_ms:.1f}ms  |  Faces: {len(faces)}",
            f"Time: {time_display}",
        ]

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Scale bboxes to resized frame
        if faces:
            h_orig, w_orig = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            if w_orig > 0 and h_orig > 0:
                sx = FRAME_WIDTH / w_orig
                sy = FRAME_HEIGHT / h_orig
                for f in faces:
                    x1, y1, x2, y2 = f["bbox"]
                    f["bbox"] = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

        frame = draw_results(frame, faces, stats)

        cv2.imshow("Face Recognition Test", frame)

        # ── Frame pacing ──────────────────────────────────────
        elapsed = time.perf_counter() - t0
        remaining = target_duration - elapsed
        wait_ms = max(1, int(remaining * 1000)) if remaining > 0 else 1

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed – stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
