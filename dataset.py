# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class imageDataset(Dataset):
    def __init__(self, real_dir, tampered_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        valid_extensions = [".jpg", ".jpeg", ".png"]

        for fname in os.listdir(real_dir):
            if any(fname.lower().endswith(ext) for ext in valid_extensions):
                self.samples.append(os.path.join(real_dir, fname))
                self.labels.append(0) # label 0 = real/original

        for fname in os.listdir(tampered_dir):
            if any(fname.lower().endswith(ext) for ext in valid_extensions):
                self.samples.append(os.path.join(tampered_dir, fname))
                self.labels.append(1) # label 1 = tampered



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
