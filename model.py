import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import torch.optim as optim
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

def detect_image(image):
    return 1

    

