import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split

class WDPTFrameDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)

        # drop unlabeled frames
        self.df = self.df[self.df["label"] != -1].reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.root_dir,
            row["video"],
            row["filename"]
        )

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

csv_path = "/Users/sanviadmin/Desktop/IndependentResearchProject/WDPT_dataset/labels.csv"
root_dir = "/Users/sanviadmin/Desktop/IndependentResearchProject/WDPT_dataset/roi_1"

df = pd.read_csv(csv_path)
df = df[df["label"] != -1]

videos = df["video"].unique()

train_videos, val_videos = train_test_split(
    videos, test_size=0.25, random_state=42
)

train_df = df[df["video"].isin(train_videos)]
val_df   = df[df["video"].isin(val_videos)]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

df = pd.read_csv(csv_path)
df = df[df["label"] != -1]

videos = df["video"].unique()

train_videos, val_videos = train_test_split(
    videos, test_size=0.25, random_state=42
)

train_df = df[df["video"].isin(train_videos)]
val_df   = df[df["video"].isin(val_videos)]

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = WDPTFrameDataset("train.csv", root_dir, transform)
val_dataset   = WDPTFrameDataset("val.csv", root_dir, transform)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_acc = eval_one_epoch(model, val_loader)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.3f} | "
        f"Val Acc: {val_acc:.3f}"
    )
# After training
torch.save(model.state_dict(), "resnet_wdpt.pth")

# 1. Initialize the model architecture
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # same as training
model = model.to(device)

# 2. Load saved weights
model.load_state_dict(torch.load("resnet_wdpt.pth", map_location=device))
model.eval()  # important! sets model to evaluation mode

from PIL import Image
from torchvision import transforms
import torch

# Define the same transform used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Example images
image_paths = ["/Users/sanviadmin/Desktop/IndependentResearchProject/Test_dataset/roi_1/video_001/frame_0194.jpg", "/Users/sanviadmin/Desktop/IndependentResearchProject/Test_dataset/roi_1/video_001/frame_0659.jpg", "/Users/sanviadmin/Desktop/IndependentResearchProject/Test_dataset/roi_1/video_003/frame_0302.jpg"]

for path in image_paths:
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension
    image = image.to(device)

    with torch.no_grad():  # no gradients needed
        output = model(image)
        pred = output.argmax(dim=1).item()  # predicted class (0 or 1)
        print(f"{path} -> {pred}")
