from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional, Tuple

import numpy as np
from annoy import AnnoyIndex
from insightface.app import FaceAnalysis


class RecognitionService:
    def __init__(
        self,
        face_data_dir: str,
        annoy_index_path: str,
        mapping_path: str,
        embedding_dim: int,
        tree: int,
    ) -> None:
        self.face_data_dir = face_data_dir
        self.annoy_index_path = annoy_index_path
        self.mapping_path = mapping_path
        self.embedding_dim = embedding_dim
        self.tree = tree

        self.face_app: Optional[FaceAnalysis] = None
        self.annoy_index: Optional[AnnoyIndex] = None
        self.idx2name: dict[str, str] = {}

    def load_assets(self, status_callback: Callable[[str], None]) -> None:
        status_callback("Thông báo: Đang tải mô hình nhận diện...")
        threading.Thread(target=self._init_recognition, args=(status_callback,), daemon=True).start()

    def _init_recognition(self, status_callback: Callable[[str], None]) -> None:
        self.face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self.face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=(320, 320))

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

        status_callback(message)

    def build_face(self, on_complete: Optional[Callable[[str], None]] = None) -> None:
        files = [f for f in os.listdir(self.face_data_dir) if f.endswith(".npy")]
        if not files:
            return

        ann = AnnoyIndex(self.embedding_dim, "angular")
        idx2name: dict[int, str] = {}
        idx = 0

        for file_name in files:
            name = file_name.replace(".npy", "")
            data = np.load(os.path.join(self.face_data_dir, file_name))
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

    def recognize(self, embedding: np.ndarray, sim_threshold: float) -> Tuple[str, float]:
        if self.annoy_index is None or self.annoy_index.get_n_items() == 0:
            return "Unknown", 0.0

        idx, dist = self.annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)
        if not dist:
            return "Unknown", 0.0

        sim = 1 - (dist[0] ** 2) / 2
        if sim >= sim_threshold:
            return self.idx2name.get(str(idx[0]), "Unknown"), sim
        return "Unknown", sim
