import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
import shutil
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split

# initialize the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # same as training
model = model.to(device)

# oad saved weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device)
    m.eval()
    return m

model_sand = load_model("resnet_wdpt.pth")
model_topsoil = load_model("resnet_wdpt_topsoil.pth")



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

if __name__ == '__main__':
    img_path = "/Users/sanviadmin/Documents/GitHub/wdpt_rpi/frame_inspection/frame_0860.jpg"
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
            pred_sand = model_sand(inp).argmax(dim=1).item()
            pred_topsoil = model_topsoil(inp).argmax(dim=1).item()

    print(f"{img_path} -> sand:{pred_sand}, topsoil:{pred_topsoil}")
        # if either says 0 → final 0
    final_pred = min(pred_sand, pred_topsoil)