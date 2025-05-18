from rembg import remove
from PIL import Image
import os
import io
from torchvision import transforms


from model import predict_image

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

def process_image(image, view):
    image = image.convert("RGB")
    clean_img = remove_bg(image)

    label, confidence = predict_image(clean_img)
    print("here")

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