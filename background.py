from rembg import remove
from PIL import Image
import os
import io

from model import detect_image


def process_image(image, view):
    clean_img = remove_bg(image)

    label, confidence = detect_image(clean_img)

    if label == 1:
        result = f"Detected AI-generated content:  {confidence:.2f}% confidence"
    else:
        result = f"No AI-generated content detected: {confidence:.2f}% confidence"

    return clean_img, result


def remove_bg(image):

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    temp = remove(img_bytes)
    clean_img = Image.open(io.BytesIO(temp)).convert("RGB")
    return clean_img