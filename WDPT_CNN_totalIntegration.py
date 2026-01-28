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

# 1. Initialize the model architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # same as training
model = model.to(device)

# 2. Load saved weights
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

import cv2
import os
def extract_roi_frames(video_path, roi_coords=(713, 248, 1150, 452), output_folder="frame_inspection", fps_out=5):
#def extract_roi_frames(video_path, roi_coords=(636, 233, 956, 450), output_folder="frame_inspection", fps_out=5):
    """
    Extract frames from a video at a specific FPS and save them as images in a folder.
    Only saves the ROI.
    """
    delete_inspection_folder(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_out)  # grab every Nth frame

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Crop ROI
            x1, y1, x2, y2 = roi_coords
            roi = frame[y1:y2, x1:x2]
            # Save frame
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), roi)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")
    return output_folder

def delete_inspection_folder(folder_name="frame_inspection"):
    # Get the absolute path to ensure we are looking in the right place
    folder_path = os.path.abspath(folder_name)
    
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted: {folder_path}")
        except Exception as e:
            print(f"Error while deleting folder: {e}")
    else:
        print(f"The folder '{folder_name}' does not exist.")

import os

def get_last_frame_from_folder(folder_path):
    files = sorted(os.listdir(folder_path))
    if not files:
        raise ValueError("No frames found in folder")
    last_frame_path = os.path.join(folder_path, files[-1])
    return last_frame_path

def classify_last_frame_from_folder(folder_path):
    last_frame_path = get_last_frame_from_folder(folder_path)
    image_paths = [last_frame_path]

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_sand = model_sand(image).argmax(dim=1).item()
            pred_topsoil = model_topsoil(image).argmax(dim=1).item()

        print(f"{path} -> sand:{pred_sand}, topsoil:{pred_topsoil}")

        # YOUR RULE: if either says 0 → final 0
        final_pred = min(pred_sand, pred_topsoil)

    return final_pred

def start_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Background Subtractor with a longer 'memory'
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False) #looks 300 frames back to see what's considered permanent background and 16 is threshold telling it how much pixel must change color before its considered motion
    
    start_time = None
    end_time = None
    frame_count = 0
    activity_log = []
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        roi = frame[313:653, 658:1005]
        # 1. Now, perform all detection only on the 'roi' instead of the 'frame'
        fg_mask = backSub.apply(roi)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel) # Removes tiny noise
        
        # Calculate current 'Activity Score' (number of changing pixels)
        activity_score = np.count_nonzero(fg_mask)
        activity_log.append(activity_score)
        
        # 2. Program "Thinking" Logic
        status = "IDLE: Waiting for drop"
        color = (200, 200, 200) # Gray
        
        if start_time is None and frame_count > 1:
            if activity_score > 1000: # Landing Threshold
                start_time = frame_count / fps
                status = "START DETECTED: Water Landed"
                color = (0, 255, 0) # Green
                print(f"Status: '{status}' Frame count '{frame_count}'.")
                return start_time


def analyze_backwards(video_path, x1, y1, x2, y2, output_dir="reverse_analysis"):
    delete_inspection_folder(output_dir)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    # 1. Load all frames into memory (for a 50s video, this is manageable)
    while True:
        ret, frame = cap.read()
        if not ret: break
        # Crop to your ROI immediately to save memory
        roi = frame[y1:y2, x1:x2].copy()
        frames.append(roi)
    cap.release()

    total_frames = len(frames)
    # The last frame is our 'Perfect Absorbed' reference
    final_reference = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    final_reference = cv2.GaussianBlur(final_reference, (5, 5), 0)

    detected_end_frame_idx = None

    # 2. Iterate backwards from the end
    for i in range(total_frames - 1, -1, -1):
        current_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        # Calculate absolute difference between current frame and the final reference
        diff = cv2.absdiff(current_gray, final_reference)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        diff_score = np.count_nonzero(thresh)

        # 3. Detection: The first time we see a "blob" (the droplet still existing)
        # going backwards, that is the exact moment absorption finished.
        if diff_score > 50:  # Adjust this 'Difference' threshold
            detected_end_frame_idx = i
            end_time = i / fps
            cv2.imwrite(f"{output_dir}/detected_end_point.jpg", frames[i])
            print(f"End point identified at frame {i} ({end_time:.2f}s)")
            return end_time

video_path = "/Users/sanviadmin/Desktop/IndependentResearchProject/SampleVideos/val/video_13.mov"
#roi_coords = (636, 233, 956, 450)
roi_coords = (713, 248, 1150, 452)
# 1. Extract ROI frames at 5 FPS
frame_folder = extract_roi_frames(video_path, roi_coords, output_folder="frame_inspection", fps_out=5)

# 2. Run CNN on last frame
majority_pred = classify_last_frame_from_folder(frame_folder)
print(f"ML-based absorption check: {majority_pred} -> {'Not absorbed' if majority_pred==0 else 'Absorbed'}")

# 3. Only run backward detection if absorption occurred
if majority_pred == 1:
    end_time = analyze_backwards(video_path, *roi_coords)
    start_time = start_detection(video_path)
    print(f"Start: {start_time:.2f}s | Reverse End: {end_time:.2f}s | WDPT: {end_time-start_time:.2f}s")
else:
    end_time = None
    print("Absorption not detected, skipping end-time detection")
