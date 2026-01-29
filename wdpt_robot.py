import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import time

TARGET_FPS = 5        # frames per second to STORE
MAX_FRAMES = 700      # hard safety cap


# =========================
# DEVICE
# =========================
device = torch.device("cpu")  # Pi 5 = CPU only

# =========================
# LOAD MODELS (TorchScript)
# =========================
# Sand model
model_sand = resnet18(weights=ResNet18_Weights.DEFAULT)
model_sand.fc = nn.Linear(model_sand.fc.in_features, 2)
model_sand.load_state_dict(torch.load("resnet_wdpt.pth", map_location=device))
model_sand.to(device)
model_sand.eval()

# Topsoil model
model_topsoil = resnet18(weights=ResNet18_Weights.DEFAULT)
model_topsoil.fc = nn.Linear(model_topsoil.fc.in_features, 2)
model_topsoil.load_state_dict(torch.load("resnet_wdpt_topsoil.pth", map_location=device))
model_topsoil.to(device)
model_topsoil.eval()

# =========================
# TRANSFORM (INFERENCE ONLY)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# CLASSIFY NUMPY FRAME
# =========================
def classify_frame_np(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sand = model_sand(image).argmax(1).item()
        topsoil = model_topsoil(image).argmax(1).item()

    return min(sand, topsoil)  # YOUR RULE

# =========================
# ROI (FROM YOUR CODE)
# =========================
#ROI = (713, 248, 1150, 452)  # x1,y1,x2,y2
ROI = (743, 259, 1072, 476)  # x1,y1,x2,y2

# =========================
# START DETECTION (WATER DROP)
# =========================
def detect_start(cap, backSub):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
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

# =========================
# BACKWARD ABSORPTION DETECT
# =========================
def detect_end(frames, fps):
    final_ref = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    final_ref = cv2.GaussianBlur(final_ref, (5,5), 0)

    for i in range(len(frames)-1, -1, -1):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        diff = cv2.absdiff(gray, final_ref)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((5,5), np.uint8)
        )

        if np.count_nonzero(thresh) > 50:
            end_time = i / fps
            print(f"[END] Absorption finished at {end_time:.2f}s")
            return end_time

    return None

# =========================
# MAIN LOOP
# =========================
def run_wdpt():
    cap = cv2.VideoCapture(0)  # Pi camera / USB cam
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=16, detectShadows=False
    )

    print("Waiting for water drop...")
    start_time = detect_start(cap, backSub)

    # =========================
    # RECORD AFTER DROP
    # =========================
    frames = []
    record_seconds = 20  # 2 minutes
    start_record = time.time()

    frame_count = 0
    frame_interval = int(fps / TARGET_FPS) or 1

    while time.time() - start_record < record_seconds:

        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Only store every Nth frame (≈5 FPS)
        if frame_count % frame_interval == 0:
            x1, y1, x2, y2 = ROI
            roi = frame[y1:y2, x1:x2].copy()
            frames.append(roi)

        # Hard safety stop
        if len(frames) >= MAX_FRAMES:
            print("[WARN] Max frame buffer reached")
            break


    cap.release()

    # =========================
    # ML CLASSIFICATION
    # =========================
    final_frame = frames[-1]
    ml_pred = classify_frame_np(final_frame)

    print(
        f"[ML] Absorption check: "
        f"{'Absorbed' if ml_pred == 1 else 'Not absorbed'}"
    )

    if ml_pred == 1:
        end_time = detect_end(frames, fps)
        if end_time is not None:
            wdpt = end_time - start_time
            print(f"[RESULT] WDPT = {wdpt:.2f} seconds")
            return wdpt

    print("[RESULT] Absorption not detected")
    return None

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_wdpt()
