from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from annoy import AnnoyIndex
from insightface.app import FaceAnalysis
import onnxruntime as ort

from app.config import FRAME_HEIGHT, FRAME_WIDTH



class RecognitionService:
    """Wraps InsightFace + Annoy for face recognition."""

    def __init__(
        self,
        face_data_dir: str,
        annoy_index_path: str,
        mapping_path: str,
        embedding_dim: int = 512,
        tree: int = 50,
        sim_threshold: float = 0.45,
    ) -> None:
        self.face_data_dir = face_data_dir
        self.annoy_index_path = annoy_index_path
        self.mapping_path = mapping_path
        self.embedding_dim = embedding_dim
        self.tree = tree
        self.sim_threshold = sim_threshold

        self.face_app: Optional[FaceAnalysis] = None
        self.annoy_index: Optional[AnnoyIndex] = None
        self.idx2name: dict[str, str] = {}
        
        
        
        self.mask_session = ort.InferenceSession("models/mask_detector.onnx", providers=['CPUExecutionProvider'])
        self.mask_input_name = self.mask_session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_assets(self, status_callback: Callable[[str], None]) -> None:
        """Kick off a background thread to load the model + index."""
        status_callback("Thông báo: Đang tải mô hình nhận diện...")
        threading.Thread(target=self._init_recognition, args=(status_callback,), daemon=True).start()

    def _init_recognition(self, status_callback: Callable[[str], None]) -> None:
        self.face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.face_app.prepare(ctx_id=0, det_thresh=0.45, det_size=(FRAME_WIDTH, FRAME_HEIGHT))

        if os.path.exists(self.annoy_index_path) and os.path.exists(self.mapping_path):
            self.annoy_index = AnnoyIndex(self.embedding_dim, "angular")
            self.annoy_index.load(self.annoy_index_path)
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.idx2name = json.load(f)
            unique_faces = set(self.idx2name.values())
            message = f"Thông báo: Đã tải {len(unique_faces)} khuôn mặt đã biết."
        else:
            self.annoy_index = None
            self.idx2name = {}
            message = "Thông báo: Chưa có dữ liệu khuôn mặt."

        print(message)
        status_callback(message)

    # ------------------------------------------------------------------
    # Build / rebuild index
    # ------------------------------------------------------------------

    def build_face(self, on_complete: Optional[Callable[[str], None]] = None) -> None:
        """Rebuild the Annoy index from *.npy files, then reload assets."""
        files = [f for f in os.listdir(self.face_data_dir) if f.endswith(".npy")]
        if not files:
            return

        ann = AnnoyIndex(self.embedding_dim, "angular")
        idx2name: dict[int, str] = {}
        idx = 0

        for file_name in files:
            name = file_name.replace(".npy", "")
            data = np.load(os.path.join(self.face_data_dir, file_name))
            if len(data) == 0:
                print(f"Warning: No data found in {file_name}")
                continue
            for vector in data:
                ann.add_item(idx, vector)
                idx2name[idx] = name
                idx += 1

        ann.build(self.tree)
        if self.annoy_index is not None:
            self.annoy_index.unload()
        ann.save(self.annoy_index_path)

        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(idx2name, f, ensure_ascii=False, indent=2)

        callback = on_complete or (lambda _: None)
        self.load_assets(callback)

    # ------------------------------------------------------------------
    # Recognize
    # ------------------------------------------------------------------

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Return (name, similarity) for a single embedding."""
        if self.annoy_index is None or self.annoy_index.get_n_items() == 0:
            return "Unknown", 0.0

        idx, dist = self.annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)
        if not dist:
            return "Unknown", 0.0

        sim = 1 - (dist[0] ** 2) / 2
        if sim >= self.sim_threshold:
            return self.idx2name.get(str(idx[0]), "Unknown"), sim
        return "Unknown", sim
    
    # ------------------------------------------------------------------
    # Mask detection
    # ------------------------------------------------------------------
    def check_mask(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "Unknown", 0
        
        # 1. Resize về đúng 128x128 theo yêu cầu của model (Index 1: 128, Index 2: 128)
        img = cv2.resize(face_roi, (128, 128))
        
        # 2. Chuyển sang RGB (đa số model deep learning dùng RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize về khoảng [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 4. Thêm chiều Batch (Index 0) để thành (1, 128, 128, 3)
        # Không dùng np.transpose vì model mong đợi số 3 ở cuối (Index 3)
        img = np.expand_dims(img, axis=0)

        # 5. Chạy Inference
        preds = self.mask_session.run(None, {self.mask_input_name: img})[0] # Lấy mảng kết quả đầu tiên
        
        # Lấy index có xác suất cao nhất
        # Thường preds sẽ có dạng [[prob_0, prob_1]]
        idx = np.argmax(preds)
        conf = preds[0][idx] # Lấy giá trị xác suất cụ thể
        
        # Thử nghiệm: Nếu đeo mask mà báo No Mask thì đảo ngược 0 và 1 ở đây
        label = "Mask" if idx == 0 else "No Mask"
        
        return label, conf
