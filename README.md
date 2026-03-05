# Face Recognition UI (PySide6)

Giao diện mẫu điểm danh khuôn mặt được dựng bằng PySide6 theo hình tham khảo.

## Chạy ứng dụng

1. Tạo môi trường ảo và cài thư viện:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Chạy ứng dụng:

```powershell
python app\main.py
```

## Ghi chú
- Đây là giao diện tĩnh để mô phỏng bố cục. Bạn có thể tích hợp luồng camera và nhận diện sau.
- Đổi `RTSP_BACKEND` trong `checkin\face_webcam_gui.py` nếu cần dùng RTSP.
