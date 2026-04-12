"""
OpenCV text rendering and image analysis helpers – framework-agnostic.
"""

import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_putText_utf8(img, text, position, font_path, font_size, color):
    """
    Draw UTF-8 text (supports Vietnamese diacritics) on an OpenCV image.

    Args:
        img (numpy.ndarray): BGR image.
        text (str): Text to render.
        position (tuple): (x, y) of the bottom-left corner.
        font_path (str): Path to a .ttf font file.
        font_size (int): Font size in pixels.
        color (tuple): BGR colour.

    Returns:
        numpy.ndarray: Image with text drawn.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size, encoding="unic")
    draw.text(position, text, font=font, fill=color[::-1])  # BGR→RGB
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def check_blur_laplacian(frame, threshold: float = 3.0):
    """
    Check whether an image is blurry using Laplacian variance.

    Returns:
        (is_blur, variance)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance < threshold, variance

def draw_corner_bbox(img, bbox, labels, track_id=None, thickness=2, ratio=0.2):
    # 1. Xác định màu sắc tĩnh (BGR)
    # Xanh lá (Green) cho người quen, Đỏ (Red) cho người lạ
    name, detect_score = labels
    is_known = name != 'Unknown'
    main_color = (0, 255, 0) if is_known else (0, 0, 255) # Xanh lá hoặc đỏ
    shadow_color = (255, 255, 255) # Màu trắng cho lớp đổ bóng

     # 2. Lấy tọa độ và tính độ dài góc
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    corner_len = int(min(w, h) * ratio)

    # 3. Hàm phụ để vẽ 8 đoạn thẳng (để tránh lặp lại code)
    def draw_corners(target_img, color, thick):
        # Top-Left
        cv2.line(target_img, (x1, y1), (x1 + corner_len, y1), color, thick)
        cv2.line(target_img, (x1, y1), (x1, y1 + corner_len), color, thick)
        # Top-Right
        cv2.line(target_img, (x2, y1), (x2 - corner_len, y1), color, thick)
        cv2.line(target_img, (x2, y1), (x2, y1 + corner_len), color, thick)
        # Bottom-Left
        cv2.line(target_img, (x1, y2), (x1 + corner_len, y2), color, thick)
        cv2.line(target_img, (x1, y2), (x1, y2 - corner_len), color, thick)
        # Bottom-Right
        cv2.line(target_img, (x2, y2), (x2 - corner_len, y2), color, thick)
        cv2.line(target_img, (x2, y2), (x2, y2 - corner_len), color, thick)

    # VẼ LỚP ĐỔ BÓNG (Vẽ trước, dày hơn)
    # Tăng thickness lên +2 hoặc +4 tùy độ phân giải 4K
    draw_corners(img, shadow_color, thickness + 1)

    # VẼ LỚP MÀU CHÍNH (Vẽ đè lên)
    draw_corners(img, main_color, thickness)

    # 4. Hiển thị tên với bóng chữ trắng
    if is_known:
        text = str(name + f" - {detect_score:.2f}")
        if track_id is not None:
            text = f"{track_id} - {text}"
        # Vẽ bóng chữ trắng phía dưới
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow_color, thickness + 1)
        # Vẽ chữ chính màu xanh/vàng
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, thickness - 1)
    else:
        text = str(f"{detect_score:.2f}")
        if track_id is not None:
            text = f"{track_id} - {text}"
        # Vẽ bóng chữ trắng phía dưới
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, shadow_color, thickness + 1)
        # Vẽ chữ chính màu xanh/vàng
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, thickness - 1)