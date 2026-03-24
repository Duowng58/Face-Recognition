"""
OpenCV text rendering and image analysis helpers – framework-agnostic.
"""

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
