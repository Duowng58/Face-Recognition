# import os
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# import time

# # ---------- 1. Load cơ sở dữ liệu embeddings ----------
# def load_face_database(folder="face_data"):
#     database = {}
#     for filename in os.listdir(folder):
#         if filename.endswith(".npy"):
#             name = filename.replace(".npy", "")
#             path = os.path.join(folder, filename)
#             embeddings = np.load(path)  # shape: (5, 512)
#             database[name] = embeddings
#     return database

# # ---------- 2. Tính cosine similarity ----------
# def recognize_face(face_embedding, database, threshold=0.5):
#     max_score = -1
#     identity = "Unknown"

#     for name, stored_embeddings in database.items():
#         sims = cosine_similarity([face_embedding], stored_embeddings)
#         score = np.max(sims)  # Lấy điểm cao nhất trong 5 góc

#         if score > max_score and score > threshold:
#             max_score = score
#             identity = name

#     return identity, max_score

# # ---------- 3. Khởi tạo mô hình ----------
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0)

# # ---------- 4. Load database ----------
# database = load_face_database()
# print(f"✅ Đã load {len(database)} người từ face_data/")

# # ---------- 5. Mở webcam / video ----------
# cap = cv2.VideoCapture(0)

# # Cài đặt FPS và frame skipping
# target_fps = 30
# frame_interval = 1.0 / target_fps
# last_frame_time = 0
# skip_frames = 3  # Số frame bỏ qua giữa mỗi lần xử lý
# frame_count = 0

# # Biến lưu kết quả nhận diện cuối cùng
# last_results = []

# while True:
#     # Kiểm tra thời gian để duy trì FPS ổn định
#     current_time = time.time()
#     elapsed = current_time - last_frame_time
    
#     if elapsed < frame_interval:
#         continue
    
#     last_frame_time = current_time
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for mirror effect
#     frame = cv2.flip(frame, 1)
    
#     frame_count += 1
#     process_this_frame = frame_count % skip_frames == 0
    
#     if process_this_frame:
#         # Xử lý nhận diện khuôn mặt
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = app.get(frame_rgb)
        
#         # Cập nhật kết quả mới
#         last_results = []
#         for face in faces:
#             bbox = face.bbox.astype(int)
#             name, score = recognize_face(face.embedding, database)
#             last_results.append((bbox, name, score))
    
#     # Vẽ kết quả cuối cùng lên frame
#     for bbox, name, score in last_results:
#         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
#         label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
#         cv2.putText(frame, label, (bbox[0], bbox[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
#     # Hiển thị FPS
#     # fps = 1.0 / elapsed if elapsed > 0 else 0
#     # cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

#     cv2.imshow("Live Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import os
import cv2
import numpy as np
import time
import json
import threading
from insightface.app import FaceAnalysis
from annoy import AnnoyIndex

# ---------- Cấu hình ----------
ANNOY_INDEX_PATH = "face_data/face_index.ann"
MAPPING_PATH = "face_data/image_paths.json"
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.6

# Biến toàn cục cho đa luồng
annoy_index = None
idx2name = None
face_app = None

# Event đồng bộ hóa
annoy_ready = threading.Event()
model_ready = threading.Event()

# ---------- Thread: Load Annoy ----------
def load_annoy():
    global annoy_index, idx2name
    if not os.path.exists(ANNOY_INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print("[❌] Không tìm thấy Annoy Index hoặc mapping. Vui lòng chạy build_face.py trước.")
        exit(1)

    print("[📦] Đang load Annoy Index...")
    annoy_index = AnnoyIndex(EMBEDDING_DIM, 'angular')
    annoy_index.load(ANNOY_INDEX_PATH)

    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        idx2name = json.load(f)

    print(f"[✅] Đã load Annoy Index ({len(idx2name)} người)")
    annoy_ready.set()  # báo luồng chính rằng đã xong

# ---------- Thread: Khởi tạo model InsightFace ----------
def load_model():
    global face_app
    print("[🚀] Đang khởi tạo mô hình InsightFace...")
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0)
    print("[✅] Đã khởi tạo mô hình.")
    model_ready.set()

# ---------- Hàm nhận diện qua Annoy ----------
def recognize_with_annoy(embedding, annoy_index, idx2name, threshold=0.5):
    idx, dist = annoy_index.get_nns_by_vector(embedding, 1, include_distances=True)
    if not dist:
        return "Unknown", 0.0
    sim = 1 - (dist[0] ** 2) / 2
    if sim > threshold:
        return idx2name.get(str(idx[0]), "Unknown"), sim
    return "Unknown", sim

# ---------- Chạy 2 luồng khởi tạo song song ----------
t1 = threading.Thread(target=load_annoy)
t2 = threading.Thread(target=load_model)

t1.start()
t2.start()

print("[⏳] Đang chuẩn bị hệ thống...")

# Chờ cả 2 luồng xong
annoy_ready.wait()
model_ready.wait()

print("[✅] Hệ thống đã sẵn sàng. Bắt đầu nhận diện!")

# ---------- Mở webcam ----------
cap = cv2.VideoCapture(0)
target_fps = 30
frame_interval = 1.0 / target_fps
last_frame_time = 0
skip_frames = 2
frame_count = 0
last_results = []

print("[🎥] Nhấn 'q' để thoát")

while True:
    current_time = time.time()
    elapsed = current_time - last_frame_time

    if elapsed < frame_interval:
        continue

    last_frame_time = current_time
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    process_this_frame = frame_count % skip_frames == 0

    if process_this_frame:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(frame_rgb)
        last_results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            name, score = recognize_with_annoy(face.embedding, annoy_index, idx2name, SIMILARITY_THRESHOLD)
            last_results.append((bbox, name, score))

    # Vẽ kết quả lên frame
    for bbox, name, score in last_results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Live Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


