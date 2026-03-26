from insightface.app import FaceAnalysis
import os
import onnxruntime as ort
import logging
import cv2
import numpy as np

ort.set_default_logger_severity(0) # 0 là Verbose - hiện tất cả

cache_path = os.path.abspath("./trt_cache")
# 1. Cấu hình riêng cho TensorRT
trt_options = {
    'device_id': 0,
    'trt_max_workspace_size': 1 << 30,
    'trt_fp16_enable': True,
    'trt_engine_cache_enable': True,
    'trt_engine_cache_path': cache_path,
}

# 2. Cấu hình riêng cho CUDA (Dùng khi TRT không hỗ trợ một số layer)
cuda_options = {
    'device_id': 0,
    'cudnn_conv_algo_search': 'DEFAULT', # Chuyển từ EXHAUSTIVE sang DEFAULT để nhanh hơn
}

# 3. Đưa vào FaceAnalysis đúng cách
app = FaceAnalysis(
    name="buffalo_s",
    root="my_models",
    providers=[
        ('TensorrtExecutionProvider', trt_options),
        #('CUDAExecutionProvider', cuda_options),
        #'CPUExecutionProvider'
    ],
    allowed_modules=["detection", "recognition"]
)
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640,640))

# 2. Đọc ảnh từ file
img_path = "" # Thay bằng đường dẫn ảnh của bạn
img = cv2.imread(img_path)

# 3. Đọc video
video_path = "" # Thay bằng đường dẫn video của bạn
video = cv2.VideoCapture(video_path)
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    # Xử lý frame ở đây
    begin_time = cv2.getTickCount()
    faces = app.get(frame)
    end_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (end_time - begin_time)
    print(f"  - FPS: {fps:.2f}")
    for i, face in enumerate(faces):
        print(f"  - Mặt {i+1}: Box {face.bbox.astype(int)}, Prob: {face.det_score:.2f}, FPS: {fps:.2f}")

if img is None:
    print("❌ Không tìm thấy file ảnh!")
else:
    print(f"📸 Đang test ảnh: {img_path}, Size: {img.shape}, Type: {img.dtype}")
    
    # KHÔNG dùng .astype(float16) ở đây vì OpenCV sẽ lỗi resize
    try:
        # InsightFace sẽ tự xử lý chuyển màu BGR -> RGB và resize nội bộ
        faces = app.get(img)
        
        print(f"✅ Thành công! Tìm thấy: {len(faces)} khuôn mặt.")
        for i, face in enumerate(faces):
            print(f"  - Mặt {i+1}: Box {face.bbox.astype(int)}, Prob: {face.det_score:.2f}")
            if face.embedding is not None:
                print(f"    Vector Embedding: {face.embedding.shape}")
                
    except Exception as e:
        print(f"❌ Lỗi: {e}")