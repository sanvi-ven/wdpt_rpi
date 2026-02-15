import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time
import serial
import os
import shutil

TARGET_FPS = 1        # frames per second to store
MAX_FRAMES = 700      # hard safety cap


device = torch.device("cpu")  # Pi 5 = CPU only

# load models
def load_model(weights_path):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device)
    m.eval()
    return m

# sand model
model_sand = load_model("resnet_wdpt.pth")

# topsoil model  
model_topsoil = load_model("resnet_wdpt_topsoil.pth")

# transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Match working file
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# classify numpy frame

def delete_inspection_folder(folder_name="frame_inspection"):
    # get file absolute path to ensure we are looking in the right place
    folder_path = os.path.abspath(folder_name)
    
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted: {folder_path}")
        except Exception as e:
            print(f"Error while deleting folder: {e}")
    else:
        print(f"The folder '{folder_name}' does not exist.")

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

        # if either says 0 → final 0
        final_pred = min(pred_sand, pred_topsoil)

    return final_pred

def classify_frame_np(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sand = model_sand(image).argmax(1).item()
        topsoil = model_topsoil(image).argmax(1).item()

    return min(sand, topsoil) 

# ROI
#ROI = (713, 248, 1150, 452)  # x1,y1,x2,y2
#ROI = (684, 185, 984, 377)  # x1,y1,x2,y2
ROI = (1700, 400, 2250, 800)  # x1,y1,x2,y2

# start detection (water drop)
def detect_start(cap, backSub):
    fps = 1 #cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        x1, y1, x2, y2 = ROI
        roi = frame[y1:y2, x1:x2]

        fg = backSub.apply(roi)
        fg = cv2.morphologyEx(
            fg, cv2.MORPH_OPEN, np.ones((5,5), np.uint8)
        )

        activity = np.count_nonzero(fg)

        if activity > 1000:
            start_time = frame_count / fps
            print(f"[START] Water landed at {start_time:.2f}s")
            return start_time

# backward absorption detect
def detect_end(frames, fps):
    # Filter out any None frames and ensure all frames have the same size
    valid_frames = []
    expected_shape = None
    
    for frame in frames:
        if frame is not None:
            if expected_shape is None:
                expected_shape = frame.shape
            elif frame.shape == expected_shape:
                valid_frames.append(frame)
            else:
                print(f"Warning: Frame with shape {frame.shape} doesn't match expected {expected_shape}, skipping")
    
    if len(valid_frames) < 2:
        print("Error: Not enough valid frames for analysis")
        return None
        
    frames = valid_frames
    print(f"Analyzing {len(frames)} valid frames")
    
    final_ref = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    final_ref = cv2.GaussianBlur(final_ref, (5,5), 0)

    for i in range(len(frames)-1, -1, -1):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        
        # double-check sizes before operations
        if gray.shape != final_ref.shape:
            print(f"Warning: Size mismatch at frame {i}, skipping")
            continue

        diff = cv2.absdiff(gray, final_ref)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8)
        )

        diff_score = np.count_nonzero(thresh)

        if diff_score > 350:  # match working file threshold
            print(f"Detection at frame {i}, diff_score={diff_score}, end_time={i/fps:.2f}s")
            end_time = i / fps
            print(f"[END] Absorption finished at {end_time:.2f}s")
            
            #save detection frame to reverse_analysis folder for inspection
            output_dir = "reverse_analysis"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv2.imwrite(f"{output_dir}/detected_end_point.jpg", frames[i])
            print(f"Saved detection frame to {output_dir}/detected_end_point.jpg")
            
            return end_time

    return None

# main loop
def run_wdpt():
    cap = cv2.VideoCapture(0)  # USB cam
    fps = 1 #cap.get(cv2.CAP_PROP_FPS) or 30

    # Camera warm-up period
    print("Warming up camera...")
    for i in range(30):  # Take 30 throwaway frames to let camera stabilize
        ret, frame = cap.read()
        if not ret:
            continue
    print("Camera warmed up with FPS:", fps)

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=16, detectShadows=False
    )

    print("Starting pump...")

    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)

    ser.write(b'1')  # pump ON
    time.sleep(1.5)  # wait for water to fall
    ser.write(b'0')  # pump OFF
    ser.close()

    # print("Waiting for water drop...")
    start_time = 0 # detect_start(cap, backSub)

    # record after drop
    output_folder = "frame_inspection"
    delete_inspection_folder(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    record_seconds = 20  # 30 seconds
    start_record = time.time()

    frame_count = 0
    saved_count = 0
    frame_interval = 1 # int(fps / TARGET_FPS) or 1

    while time.time() - start_record < record_seconds:

        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # only store every Nth frame (≈5 FPS)
        if frame_count % frame_interval == 0:
            x1, y1, x2, y2 = ROI
            roi = frame[y1:y2, x1:x2].copy()
            
            # save only ROI frame to disk (like working file)
            frame_name = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        # hard safety stop
        if saved_count >= MAX_FRAMES:
            print("[WARN] Max frame buffer reached")
            break


    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")

    # ml classification
    ml_pred = classify_last_frame_from_folder(output_folder)

    print(
        f"[ML] Absorption check: "
        f"{'Absorbed' if ml_pred == 1 else 'Not absorbed'}"
    )

    if ml_pred == 1:
        # Load frames from disk for detect_end with error checking
        files = sorted(os.listdir(output_folder))
        frames = []
        loaded_count = 0
        
        for file in files:
            if file.endswith('.jpg'):  # Only process JPEG files
                frame_path = os.path.join(output_folder, file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
                    loaded_count += 1
                else:
                    print(f"Warning: Failed to load {frame_path}")
        
        print(f"Loaded {loaded_count} frames for analysis")
        
        if len(frames) > 0:
            end_time = detect_end(frames, TARGET_FPS)
            if end_time is not None:
                wdpt = end_time - start_time
                print(f"[RESULT] WDPT = {wdpt:.2f} seconds")
                return wdpt
        else:
            print("Error: No valid frames loaded for analysis")

    print("[RESULT] Absorption not detected")
    return None

# run
if __name__ == "__main__":
    run_wdpt()

