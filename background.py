from rembg import remove
from PIL import Image
import os
import io
from torchvision import transforms
import requests
import openai

from model import predict_image

openai.api_key = "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"  


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

def process_image(image, view):
    image = image.convert("RGB")
    clean_img = remove_bg(image)

    label, confidence = predict_image(clean_img)
    print("here")

    ex = reason(label, confidence, view)

    if label == 1:
        result = f"Detected AI-generated content:  {confidence:.2f}% confidence"
    else:
        result = f"No AI-generated content detected: {confidence:.2f}% confidence"

    return clean_img, result, ex


def remove_bg(image):

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    temp = remove(img_bytes)
    clean_img = Image.open(io.BytesIO(temp)).convert("RGB")
    return clean_img


def reason(label, confidence, view):
    prompt = f"""you are an ai agent with a a detection model, you predicted that an image of the {view} view of a car is {'ai-generated' if label == 1 else 'real'} with {confidence:.1f}% confidence.
    give a brief explaination why you might have made this decision."""

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=60
    )
    return response.choices[0].message.content.strip()

