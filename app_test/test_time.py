import re
import datetime
import cv2

filepath = "assets/videos/record-2026-03-25-06-01-24.mkv"

# Tìm chuỗi có định dạng số-số-số...
match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})', filepath)

if match:
    time_str = match.group(1)
    start_time = datetime.datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S")
    print(f"Kết quả: {start_time}")
    

cap = cv2.VideoCapture(filepath)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Lấy vị trí thời gian hiện tại của frame (tính bằng miliseconds)
    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    # 3. Cộng thêm vào thời gian bắt đầu
    current_actual_time = start_time + datetime.timedelta(milliseconds=msec)

    # Hiển thị kết quả lên màn hình video (tùy chọn)
    time_display = current_actual_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cv2.putText(frame, time_display, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()