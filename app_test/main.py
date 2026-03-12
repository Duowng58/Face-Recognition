import cv2
import numpy as np
import onnxruntime as ort
import os
import time
mask_session = ort.InferenceSession("models/mask_detector.onnx", providers=['CPUExecutionProvider'])
input_name = mask_session.get_inputs()[0].name

def check_mask(face_roi):
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
    preds = mask_session.run(None, {input_name: img})[0] # Lấy mảng kết quả đầu tiên
    
    # Lấy index có xác suất cao nhất
    # Thường preds sẽ có dạng [[prob_0, prob_1]]
    idx = np.argmax(preds)
    conf = preds[0][idx] # Lấy giá trị xác suất cụ thể
    
    # Thử nghiệm: Nếu đeo mask mà báo No Mask thì đảo ngược 0 và 1 ở đây
    label = "Mask" if idx == 0 else "No Mask"
    
    return label, conf

# load all image trong folder app_test/images

start_time = time.time()
image_folder = "app_test/images"
images = [(cv2.imread(os.path.join(image_folder, f)), f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

for image, filename in images:
    label, conf = check_mask(image)
    print(f"File: {filename}, Label: {label}, Confidence: {conf}")
    
end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")