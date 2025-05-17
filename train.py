import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
import torch.optim as optim
import os

from dataset import imageDataset

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize images to 224x224 for ResNet model
    transforms.ToTensor()
])

dataset = imageDataset( #load images and applies transformations
    real_dir="data/real/clean_rl",
    tampered_dir="data/ai/clean_ai",
    transform=transform
)

train_size = int(0.8 * len(dataset) )
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size,test_size])


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# using pretrained ResNet model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2) #2 output classes
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# training 
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")

    
torch.save(model.state_dict(), "models/resnet18_ai.pth")
print("done")