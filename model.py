import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

from dataset import imageDataset

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Using device: {device}")

# loading trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet18_imd2020.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize images to 224x224 for ResNet model
    transforms.ToTensor()
])


def predict_image(img):
    # image = Image.open(image).convert("RGB")
    image = img.convert("RGB")

    input = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(input)

        probs = torch.softmax(out, dim=1)[0]

        label = torch.argmax(probs).item()

        confidence = probs[label].item()*100

    return label, round(confidence, 2)



