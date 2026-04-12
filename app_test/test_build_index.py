"""
Build / update Annoy face index from a video file or a folder of face images.

Usage:
    python app_test/test_build_index.py --name "Nguyen Van A" --video path/to/video.mkv
    python app_test/test_build_index.py --name "Nguyen Van A" --folder path/to/frames/
    python app_test/test_build_index.py --name "Nguyen Van A" --video path/to/video.mkv --preview
    python app_test/test_build_index.py --name "Nguyen Van A" --video path/to/video.mkv --threshold 0.75
    python app_test/test_build_index.py --rebuild   (chỉ rebuild index từ .npy đã có, không trích xuất mới)
"""

from __future__ import annotations

import argparse
import json
from math import e
import os
import sys
import time
from tracemalloc import start

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
    FACE_DATA_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAPPING_PATH,
    DETECT_SIZE
)

EMBEDDING_DIM = 512


# ─────────────────────────────────────────────────────────────
# Face-analysis model (lazy singleton)
# ─────────────────────────────────────────────────────────────
DETECT_SIZE = (320,320)
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
    return _face_app


# ─────────────────────────────────────────────────────────────
# Letterbox helpers (same logic as RecognitionService)
# ─────────────────────────────────────────────────────────────

def _letterbox_params(orig_h: int, orig_w: int):
    tw, th = DETECT_SIZE
    scale = min(tw / orig_w, th / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    offset_x = (tw - new_w) // 2
    offset_y = (th - new_h) // 2
    return scale, offset_x, offset_y


def _letterbox(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale, ox, oy = _letterbox_params(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((DETECT_SIZE[1], DETECT_SIZE[0], 3), 128, dtype=np.uint8)
    canvas[oy : oy + resized.shape[0], ox : ox + resized.shape[1]] = resized
    
    cv2.imshow("frame_small", canvas)
    cv2.waitKey(1)
    return canvas


def _map_to_original(coords, scale, ox, oy):
    return (np.array(coords) - np.array([ox, oy])) / scale


# ─────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────

def extract_embeddings_from_frame(frame: np.ndarray) -> list[np.ndarray]:
    """Return a list of normed_embedding vectors for all faces in *frame*."""
    app = get_face_app()
    h, w = frame.shape[:2]
    # faces = app.get(frame)
    # return [e.normed_embedding for e in faces]
    small = _letterbox(frame)
    scale, ox, oy = _letterbox_params(h, w)

    bboxes, kpss = app.models["detection"].detect(small)
    embeddings: list[np.ndarray] = []
    bboxes_4k: list[list[int]] = []
    
   

    if bboxes.shape[0] == 0:
        return embeddings, bboxes_4k
    
     # tìm bbox có area lớn nhất (Khuôn mặt gần camera nhất)
    max_area = 0
    max_idx = -1
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2, score = bboxes[i]
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_idx = i

    if max_idx != -1:
        try:
            
            x1_sm, y1_sm, x2_sm, y2_sm, score = bboxes[max_idx]
            
            # 2. Ánh xạ ngược về tọa độ 4K
            # Công thức: (Tọa độ ảnh nhỏ - phần bù viền đen) / tỉ lệ scale
            x1_4k = (x1_sm - ox) / scale
            y1_4k = (y1_sm - oy) / scale
            x2_4k = (x2_sm - ox) / scale
            y2_4k = (y2_sm - oy) / scale
            
            bbox_4k = [int(x1_4k), int(y1_4k), int(x2_4k), int(y2_4k)]
            bboxes_4k.append(bbox_4k)
            kps_orig = _map_to_original(kpss[max_idx], scale, ox, oy)
            face_aimg = face_align.norm_crop(frame, kps_orig)
            feat = app.models["recognition"].get_feat(face_aimg)
            normed = feat / np.linalg.norm(feat)
            embeddings.append(normed.flatten())
        except Exception as e:
            print(f"  [WARN] Không trích được khuôn mặt {max_idx}: {e}")

    return embeddings, bboxes_4k


# ─────────────────────────────────────────────────────────────
# Duplicate filter
# ─────────────────────────────────────────────────────────────

def filter_unique_embeddings(
    embeddings: list[np.ndarray], threshold: float = 0.9
) -> np.ndarray:
    """Lọc các embedding trùng lặp dựa trên cosine similarity."""
    if not embeddings:
        return np.array([])

    unique: list[np.ndarray] = [embeddings[0]]
    for emb in embeddings[1:]:
        sims = np.dot(np.array(unique), emb)
        if np.max(sims) < threshold:
            unique.append(emb)
    return np.array(unique)


# ─────────────────────────────────────────────────────────────
# Source: Video file
# ─────────────────────────────────────────────────────────────

def extract_from_video(
    video_path: str, threshold: float, preview: bool
) -> list[np.ndarray]:
    """Đọc video, trích xuất embedding từ tất cả frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không mở được video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {video_path}")
    print(f"  FPS: {fps:.1f}  |  Tổng frame: {total_frames}")

    all_embeddings: list[np.ndarray] = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        start_time = time.time()
        embs, bboxes = extract_embeddings_from_frame(frame)
        all_embeddings.extend(embs)

        if preview:
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            display = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            info = f"Frame {count}/{total_frames}  |  Faces so far: {len(all_embeddings)}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Build Index - Video", display)
            end_time = time.time()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  [INFO] Đã dừng bởi phím 'q'.")
                break
            if end_time - start_time < 1 / fps:
                time.sleep(1 / fps - (end_time - start_time))

        if count % 50 == 0:
            print(f"  Processed {count}/{total_frames} frames  |  Raw embeddings: {len(all_embeddings)}")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    print(f"  Tổng frame đã xử lý: {count}  |  Raw embeddings: {len(all_embeddings)}")
    unique = filter_unique_embeddings(all_embeddings, threshold)
    print(f"  Sau lọc trùng (threshold={threshold}): {len(unique)} embeddings")
    return list(unique) if unique.size > 0 else []


# ─────────────────────────────────────────────────────────────
# Source: Folder of images
# ─────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_from_folder(
    folder: str, threshold: float, preview: bool
) -> list[np.ndarray]:
    """Đọc tất cả ảnh trong folder, trích xuất embedding."""
    files = sorted(
        f
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )
    if not files:
        print(f"[ERROR] Không tìm thấy ảnh trong: {folder}")
        return []

    print(f"  Folder: {folder}")
    print(f"  Tổng ảnh: {len(files)}")

    all_embeddings: list[np.ndarray] = []

    for i, fname in enumerate(files, 1):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Không đọc được: {fname}")
            continue

        embs = extract_embeddings_from_frame(img)
        all_embeddings.extend(embs)

        if preview:
            display = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            info = f"Image {i}/{len(files)}  |  Faces so far: {len(all_embeddings)}"
            cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Build Index - Images", display)
            if cv2.waitKey(300) & 0xFF == ord("q"):
                print("  [INFO] Đã dừng bởi phím 'q'.")
                break

        if i % 20 == 0:
            print(f"  Processed {i}/{len(files)} images  |  Raw embeddings: {len(all_embeddings)}")

    if preview:
        cv2.destroyAllWindows()

    print(f"  Tổng ảnh đã xử lý: {len(files)}  |  Raw embeddings: {len(all_embeddings)}")
    unique = filter_unique_embeddings(all_embeddings, threshold)
    print(f"  Sau lọc trùng (threshold={threshold}): {len(unique)} embeddings")
    return list(unique) if unique.size > 0 else []


# ─────────────────────────────────────────────────────────────
# Build / Rebuild Annoy index
# ─────────────────────────────────────────────────────────────

def build_annoy_index(n_trees: int = 50) -> None:
    """Đọc tất cả .npy trong FACE_DATA_DIR và build lại Annoy index + mapping JSON."""
    files = [f for f in os.listdir(FACE_DATA_DIR) if f.endswith(".npy")]
    if not files:
        print("[WARN] Không có file .npy nào trong", FACE_DATA_DIR)
        return

    ann = AnnoyIndex(EMBEDDING_DIM, "angular")
    idx2name: dict[int, str] = {}
    idx = 0

    for fname in sorted(files):
        name = os.path.splitext(fname)[0]
        data = np.load(os.path.join(FACE_DATA_DIR, fname))
        if data.dtype == object:
            data = np.array(list(data))
        if data.ndim == 1:
            data = data.reshape(1, -1)

        for i in range(data.shape[0]):
            vec = data[i]
            if vec is None or len(vec) != EMBEDDING_DIM:
                print(f"  [WARN] Bỏ qua vector lỗi: {fname} index {i}")
                continue
            ann.add_item(idx, vec)
            idx2name[idx] = name
            idx += 1

    ann.build(n_trees)
    ann.save(ANNOY_INDEX_PATH)

    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(idx2name, f, ensure_ascii=False, indent=2)

    unique_names = set(idx2name.values())
    print(f"[OK] Annoy index built: {idx} vectors, {len(unique_names)} people")
    print(f"     Index: {ANNOY_INDEX_PATH}")
    print(f"     Mapping: {MAPPING_PATH}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build / cập nhật Annoy face index từ video hoặc folder ảnh.",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=None,
        help="Tên người (dùng làm tên file .npy). Bắt buộc khi dùng --video hoặc --folder.",
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Đường dẫn tới video file để trích xuất khuôn mặt.",
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=None,
        help="Đường dẫn tới folder chứa ảnh khuôn mặt.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.8,
        help="Ngưỡng cosine similarity để lọc embedding trùng (default: 0.8).",
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Hiển thị cửa sổ preview khi xử lý.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Chỉ rebuild Annoy index từ các .npy đã có (không trích xuất mới).",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=50,
        help="Số lượng cây cho Annoy index (default: 50).",
    )
    args = parser.parse_args()

    os.makedirs(FACE_DATA_DIR, exist_ok=True)

    # ── Rebuild only ──────────────────────────────────────────
    if args.rebuild:
        print("=" * 50)
        print("Rebuild Annoy index từ .npy đã có")
        print("=" * 50)
        build_annoy_index(n_trees=args.trees)
        return

    # ── Extract + Build ───────────────────────────────────────
    if not args.name:
        parser.error("--name là bắt buộc khi dùng --video hoặc --folder.")

    if not args.video and not args.folder:
        parser.error("Cần ít nhất --video hoặc --folder (hoặc --rebuild).")

    safe_name = args.name.strip().replace(" ", "_")
    npy_path = os.path.join(FACE_DATA_DIR, f"{safe_name}.npy")

    print("=" * 50)
    print(f"  Name:      {args.name}")
    print(f"  NPY file:  {npy_path}")
    print(f"  Threshold: {args.threshold}")
    print("=" * 50)

    t0 = time.time()

    embeddings: list[np.ndarray] = []

    if args.video:
        embeddings = extract_from_video(args.video, args.threshold, args.preview)
    elif args.folder:
        embeddings = extract_from_folder(args.folder, args.threshold, args.preview)

    if not embeddings:
        print("[WARN] Không trích xuất được embedding nào. Dừng.")
        return

    # ── Tạo hoặc replace file .npy cũ (nếu có) ──────────────────────
    if os.path.exists(npy_path):
        os.remove(npy_path)

    np.save(npy_path, embeddings)
    print(f"[OK] Saved {len(embeddings)} embeddings → {npy_path}")

    # ── Rebuild index ─────────────────────────────────────────
    print()
    build_annoy_index(n_trees=args.trees)

    elapsed = time.time() - t0
    print(f"\nHoàn thành trong {elapsed:.1f}s")


if __name__ == "__main__":
    main()
