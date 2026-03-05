import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_putText_utf8(img, text, position, font_path, font_size, color):
    """
    Vẽ chữ UTF-8 (hỗ trợ tiếng Việt có dấu) lên ảnh OpenCV

    Args:
        img (numpy.ndarray): Ảnh định dạng OpenCV (BGR)
        text (str): Nội dung cần vẽ
        position (tuple): Tọa độ (x, y) của góc dưới bên trái chữ
        font_path (str): Đường dẫn tới file font hỗ trợ tiếng Việt (VD: Arial.ttf)
        font_size (int): Cỡ chữ
        color (tuple): Màu chữ (B, G, R), VD: (0, 0, 255) là màu đỏ

    Returns:
        numpy.ndarray: Ảnh OpenCV đã được vẽ chữ
    """
    # 1. Chuyển ảnh OpenCV (BGR) sang RGB để PIL xử lý
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # 2. Tạo đối tượng vẽ trên ảnh PIL
    draw = ImageDraw.Draw(pil_img)

    # 3. Load font hỗ trợ tiếng Việt
    #    encoding='unic' giúp đọc đúng ký tự có dấu
    font = ImageFont.truetype(font_path, font_size, encoding='unic')

    # 4. Vẽ chữ lên ảnh PIL (lưu ý PIL dùng màu RGB)
    draw.text(position, text, font=font, fill=color[::-1])  # Đảo BGR -> RGB

    # 5. Chuyển ảnh PIL về lại OpenCV (RGB -> BGR)
    img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return img_result


def check_blur_laplacian(frame, threshold=3.0):
    """
    Kiểm tra độ nhòe của ảnh bằng phương pháp Laplacian variance
    
    Args:
        frame: ảnh định dạng OpenCV (BGR)
        threshold: ngưỡng phát hiện nhòe (giá trị càng thấp, ảnh càng dễ bị coi là nhòe)
    
    Returns:
        (is_blur, variance): (có bị nhòe không, giá trị variance)
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tính toán Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Tính variance
    variance = laplacian.var()
    
    # Kiểm tra ngưỡng
    is_blur = variance < threshold
    
    return is_blur, variance