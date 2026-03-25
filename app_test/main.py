import os
import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ── path setup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from scripts.config import FRAME_HEIGHT, FRAME_WIDTH, VIDEO

# 1. Khởi tạo InsightFace (nên dùng model nhẹ hơn buffalo_l nếu Jetson Nano bị treo)
app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) # Detection chạy ở 640x640 để nhanh

def get_letterbox_params(orig_h, orig_w, target_size=(640, 640)):
    tw, th = target_size
    scale = min(tw / orig_w, th / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Tính toán phần viền đen để căn giữa (offset)
    offset_x = (tw - new_w) // 2
    offset_y = (th - new_h) // 2
    
    return scale, offset_x, offset_y
def letterbox(img, target_size=(640, 640)):
    h, w = img.shape[:2]
    scale, ox, oy = get_letterbox_params(h, w, target_size)
    
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8) # Màu xám trung tính
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    
    return canvas
def map_to_original(coords, scale, ox, oy):
    # coords có shape (5, 2) cho kps hoặc (n,) cho bbox
    # Chúng ta biến ox, oy thành một mảng có hình dạng tương ứng
    offset = np.array([ox, oy])
    
    # Ép kiểu coords về numpy để tính toán nếu nó chưa phải
    coords = np.array(coords)
    
    # Phép toán: (Tọa độ - Offset) / Scale
    # Numpy sẽ tự hiểu và trừ ox cho cột 0, oy cho cột 1
    return (coords - offset) / scale
def get_high_accuracy_embeddings(frame_4k):
    # Bước 1: Tạo ảnh nhỏ để Detect (giảm tải GPU)
    h_orig, w_orig = frame_4k.shape[:2]
    input_size = (640, 640)
    frame_small = letterbox(frame_4k, input_size)
    scale, ox, oy = get_letterbox_params(h_orig, w_orig, input_size)
    # Tính tỉ lệ để map ngược lại ảnh 4K
    sw, sh = input_size
    rx, ry = w_orig / sw, h_orig / sh

    # Bước 2: Chỉ thực hiện DETECT trên ảnh nhỏ
    # Chúng ta dùng app.models['detection'] trực tiếp để tránh chạy nhận diện toàn bộ ảnh nhỏ
    bboxes, kpss = app.models['detection'].detect(frame_small)
    
    results = []
    if bboxes.shape[0] > 0:
        for i in range(bboxes.shape[0]):
            # Bước 3: Map tọa độ Box và Keypoints về 4K
            kps_4k = map_to_original(kpss[i], scale, ox, oy)
            x1_sm, y1_sm, x2_sm, y2_sm, score = bboxes[i]
    
            # 2. Ánh xạ ngược về tọa độ 4K
            # Công thức: (Tọa độ ảnh nhỏ - phần bù viền đen) / tỉ lệ scale
            x1_4k = (x1_sm - ox) / scale
            y1_4k = (y1_sm - oy) / scale
            x2_4k = (x2_sm - ox) / scale
            y2_4k = (y2_sm - oy) / scale
            
            # 3. Ép kiểu về số nguyên để vẽ OpenCV hoặc Crop
            bbox_4k = [int(x1_4k), int(y1_4k), int(x2_4k), int(y2_4k)]
            
            # Bước 4: Cắt (Crop) và Căn chỉnh (Align) từ ảnh 4K gốc
            # Đây là bước chốt để có normed_embedding chính xác nhất
            # InsightFace dùng 5 điểm landmark (kps) để xoay mặt thẳng lại
            face_aimg = face_align.norm_crop(frame_4k, kps_4k)
            
            # Bước 5: Trích xuất Embedding từ vùng ảnh nét nhất
            feat = app.models['recognition'].get_feat(face_aimg)
            normed_embedding = feat / np.linalg.norm(feat)
            
            results.append({
                'bbox': bbox_4k,
                'embedding': normed_embedding, # Dùng cái này đưa vào AnnoyIndex
                'aligned_face': face_aimg,     # Có thể dùng để hiển thị thumbnail
                "score": score
            })
            
    return results

# Thử nghiệm với luồng camera
cap = cv2.VideoCapture(VIDEO) 
count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if count % 5 != 0:
        count += 1
        continue
    begin_time = time.time()
    
    predictions = get_high_accuracy_embeddings(frame)
    # Vẽ kết quả lên ảnh 4K để hiển thị ra Qt5
    for p in predictions:
        b = p['bbox']
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(frame, p.get('name', 'Unknown') + ' - ' + f"{p.get('score', 0.0):.2f}", (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        try:
            frame[b[1]:b[1]+112, b[0]:b[0]+112] = p['aligned_face']
        except Exception as e:
            print(f"Error overlaying aligned face: {e}")

    end_time = time.time()
    print(f"Processing time: {end_time - begin_time:.2f} seconds")
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.imshow("Jetson Nano 4K Face ID", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    # time.sleep(0.01)