"""
Face recognition service – framework-agnostic.

Wraps InsightFace + Annoy for face detection, recognition, and mask detection.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from annoy import AnnoyIndex
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import onnxruntime as ort

from scripts.config import DETECT_SIZE


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

        self.mask_session = ort.InferenceSession(
            "models/mask_detector.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.mask_input_name = self.mask_session.get_inputs()[0].name
        
        trt_options = {
            'device_id': 0,
            'trt_max_workspace_size': 1 << 30,
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': os.path.abspath("./trt_cache"),
        }

        self.face_app = FaceAnalysis(
            name="buffalo_s",
            root="./my_models",
            providers=[
                ("TensorrtExecutionProvider", trt_options),
                "CUDAExecutionProvider",
            ],
            allowed_modules=["detection", "recognition"],
        )
        self.face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=DETECT_SIZE)
        
                # 2. Đọc ảnh từ file
        img_path = "vlcsnap-2026-03-10-16h43m12s822.png" # Thay bằng đường dẫn ảnh của bạn
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Không tìm thấy file ảnh!")
        else:
            print(f"📸 Đang test ảnh: {img_path}, Size: {img.shape}, Type: {img.dtype}")
            
            # KHÔNG dùng .astype(float16) ở đây vì OpenCV sẽ lỗi resize
            try:
                # InsightFace sẽ tự xử lý chuyển màu BGR -> RGB và resize nội bộ
                faces = self.face_app.get(img)
                
                print(f"✅ Thành công! Tìm thấy: {len(faces)} khuôn mặt.")
                for i, face in enumerate(faces):
                    print(f"  - Mặt {i+1}: Box {face.bbox.astype(int)}, Prob: {face.det_score:.2f}")
                    if face.embedding is not None:
                        print(f"    Vector Embedding: {face.embedding.shape}")
                        
            except Exception as e:
                print(f"❌ Lỗi: {e}")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_assets(self, status_callback: Callable[[str], None]) -> None:
        """Kick off a background thread to load the model + index."""
        status_callback("Thông báo: Đang tải mô hình nhận diện...")
        threading.Thread(
            target=self._init_recognition, args=(status_callback,), daemon=True
        ).start()

    def _init_recognition(self, status_callback: Callable[[str], None]) -> None:


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

    def get_letterbox_params(self, orig_h, orig_w):
        tw, th = DETECT_SIZE
        scale = min(tw / orig_w, th / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        
        # Tính toán phần viền đen để căn giữa (offset)
        offset_x = (tw - new_w) // 2
        offset_y = (th - new_h) // 2
        
        return scale, offset_x, offset_y
    def letterbox(self, img):
        h, w = img.shape[:2]
        scale, ox, oy = self.get_letterbox_params(h, w)
        
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        canvas = np.full((DETECT_SIZE[1], DETECT_SIZE[0], 3), 128, dtype=np.uint8) # Màu xám trung tính
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        
        return canvas
    def map_to_original(self, coords, scale, ox, oy):
        # coords có shape (5, 2) cho kps hoặc (n,) cho bbox
        # Chúng ta biến ox, oy thành một mảng có hình dạng tương ứng
        offset = np.array([ox, oy])
        
        # Ép kiểu coords về numpy để tính toán nếu nó chưa phải
        coords = np.array(coords)
        
        # Phép toán: (Tọa độ - Offset) / Scale
        # Numpy sẽ tự hiểu và trừ ox cho cột 0, oy cho cột 1
        return (coords - offset) / scale
    def get_embeddings(self, frame_4k):
        # Bước 1: Tạo ảnh nhỏ để Detect (giảm tải GPU)
        h_orig, w_orig = frame_4k.shape[:2]
        input_size = DETECT_SIZE
        frame_small = self.letterbox(frame_4k)
        scale, ox, oy = self.get_letterbox_params(h_orig, w_orig)
        # Tính tỉ lệ để map ngược lại ảnh 4K
        # sw, sh = input_size
        # rx, ry = w_orig / sw, h_orig / sh

        # Bước 2: Chỉ thực hiện DETECT trên ảnh nhỏ
        # Chúng ta dùng app.models['detection'] trực tiếp để tránh chạy nhận diện toàn bộ ảnh nhỏ
        # print('start detect')
        bboxes, kpss = self.face_app.models['detection'].detect(frame_small)
        # print('end detect')
        results = []
        try: 
            if bboxes.shape[0] > 0:
                # print('1')
                for i in range(bboxes.shape[0]):
                    # print('2')
                    # Bước 3: Map tọa độ Box và Keypoints về 4K
                    kps_4k = self.map_to_original(kpss[i], scale, ox, oy)
                    # print('3')
                    x1_sm, y1_sm, x2_sm, y2_sm, score = bboxes[i]
            
                    # 2. Ánh xạ ngược về tọa độ 4K
                    # Công thức: (Tọa độ ảnh nhỏ - phần bù viền đen) / tỉ lệ scale
                    x1_4k = (x1_sm - ox) / scale
                    y1_4k = (y1_sm - oy) / scale
                    x2_4k = (x2_sm - ox) / scale
                    y2_4k = (y2_sm - oy) / scale
                    
                    bbox_4k = [int(x1_4k), int(y1_4k), int(x2_4k), int(y2_4k)]
                    
                    # Bước 4: Cắt (Crop) và Căn chỉnh (Align) từ ảnh 4K gốc
                    # Đây là bước chốt để có normed_embedding chính xác nhất
                    # InsightFace dùng 5 điểm landmark (kps) để xoay mặt thẳng lại
                    face_aimg = face_align.norm_crop(frame_4k, kps_4k)
                    # print('4')
                    
                    # Bước 5: Trích xuất Embedding từ vùng ảnh nét nhất
                    feat = self.face_app.models['recognition'].get_feat(face_aimg)
                    normed_embedding = (feat / np.linalg.norm(feat)).flatten()
                    # print('5')
                    results.append({
                        'bbox': bbox_4k,
                        'normed_embedding': normed_embedding, # Dùng cái này đưa vào AnnoyIndex
                        'aligned_face': face_aimg,     # Có thể dùng để hiển thị thumbnail
                        "det_score": score
                    })
                    # print('6')
        except Exception as e:
            print(f"Error processing face {i}: {e}")
        # print(len(results))
                
        return results
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

            if data.dtype == object:
                data = np.array(list(data))

            if data.ndim == 1:
                data = data.reshape(1, -1)

            num_vectors = data.shape[0]
            for i in range(num_vectors):
                vector = data[i]
                if vector is None or len(vector) != self.embedding_dim:
                    print(
                        f"Bỏ qua vector lỗi tại file {file_name}, index {i} "
                        f"(Length: {len(vector) if vector is not None else 0})"
                    )
                    continue
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

        idx, dist = self.annoy_index.get_nns_by_vector(
            embedding, 1, include_distances=True
        )
        if not dist:
            return "Unknown", 0.0

        sim = 1 - (dist[0] ** 2) / 2
        name = self.idx2name.get(str(idx[0]), "Unknown")
        # print(f"  [DEBUG] idx_list: {idx}, dist_list: {dist}, sim: {sim}, name: {name}")
        
        if sim >= self.sim_threshold:
            return self.idx2name.get(str(idx[0]), "Unknown"), sim
        return "Unknown", sim

    # ------------------------------------------------------------------
    # Mask detection
    # ------------------------------------------------------------------

    def check_mask(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return "Unknown", 0

        img = cv2.resize(face_roi, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = self.mask_session.run(None, {self.mask_input_name: img})[0]
        idx = np.argmax(preds)
        conf = preds[0][idx]
        label = "Mask" if idx == 0 else "No Mask"
        return label, conf
