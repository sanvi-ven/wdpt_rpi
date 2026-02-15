# Water Drop Penetration Time (WDPT) CNN Classification Model for Sand
# This script trains a ResNet18 model to classify whether water has been absorbed in sand samples

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

    #labels: 0 = not absorbed, 1 = absorbed

    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)

        # Remove any frames that weren't labeled (label = -1 means not for model)
        self.df = self.df[self.df["label"] != -1].reset_index(drop=True)

        self.root_dir = root_dir  # directory containing the video folders
        self.transform = transform  # image preprocessing transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # build the full path to the image file
        img_path = os.path.join(
            self.root_dir,
            row["video"],     # video folder name 
            row["filename"]   # frame filename 
        )

        # load image and convert to RGB 
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])  # convert label to integer

        # apply image transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# paths to the labeled data
csv_path = "/Users/sanviadmin/Desktop/IndependentResearchProject/WDPT_dataset/labels.csv"
root_dir = "/Users/sanviadmin/Desktop/IndependentResearchProject/WDPT_dataset/roi_1"

# load the dataset and remove unlabeled frames
df = pd.read_csv(csv_path)
df = df[df["label"] != -1]

# get list of unique video names for train/validation split
videos = df["video"].unique()

# split videos into training (75%) and validation (25%) sets
# using video-level split ensures frames from same video don't appear in both sets
train_videos, val_videos = train_test_split(
    videos, test_size=0.25, random_state=42
)

# create training and validation dataframes
train_df = df[df["video"].isin(train_videos)]
val_df   = df[df["video"].isin(val_videos)]

# save the splits to CSV files for reproducibility
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

# remove duplicate data splitting code (this was repeated accidentally)
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

# image preprocessing pipeline
# ResNet expects 224x224 input images with specific normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),                    # Resize to ResNet input size
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # for better generalization
    transforms.ToTensor(),                            # Convert PIL image to tensor
    transforms.Normalize(                             # Normalize using ImageNet statistics
        mean=[0.485, 0.456, 0.406],                  # RGB means
        std=[0.229, 0.224, 0.225]                    # RGB standard deviations
    )
])

# Create dataset objects for training and validation
train_dataset = WDPTFrameDataset("train.csv", root_dir, transform)
val_dataset   = WDPTFrameDataset("val.csv", root_dir, transform)

# Create data loaders for batch processing
# DataLoaders handle shuffling, batching, and parallel loading
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=0
)

# Set up device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model- ResNet18 with pretrained ImageNet weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # replace final layer for binary classification (2 classes)
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()  # standard loss for classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 0.0001

def train_one_epoch(model, loader):
    #Train the model for one epoch and return loss and accuracy.
    
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        # Mmove data to the same device as model (CPU/GPU)
        imgs, labels = imgs.to(device), labels.to(device)
        # clear gradients from previous iteration
        optimizer.zero_grad()
        outputs = model(imgs)
        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # track statistics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)  # Get predicted class (0 or 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Return average loss and accuracy for this epoch
    return total_loss / len(loader), correct / total


def eval_one_epoch(model, loader):

    model.eval()  # Set model to evaluation mode (disables dropout, batch norm updates)
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient computation for efficiency
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# Training configuration
num_epochs = 10

print("Starting training for sand absorption classification")
print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

# main training loop
for epoch in range(num_epochs):
    # train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader)
    
    # evaluate on validation set
    val_acc = eval_one_epoch(model, val_loader)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.3f} | "
        f"Val Acc: {val_acc:.3f}"
    )

# save the trained model weights for later use on Raspberry Pi
torch.save(model.state_dict(), "resnet_wdpt.pth")


