# import cv2, socket, time, serial
# from pathlib import Path
# from PIL import Image
# import torch
# from torchvision import transforms
# from torchvision.models import resnet18, ResNet18_Weights
# import torch.nn as nn
# import shutil
# import os
# import numpy as np

# # ----- SETUP DEVICE -----
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ----- LOAD MODELS -----
# def load_model(weights_path):
#     m = resnet18(weights=ResNet18_Weights.DEFAULT)
#     m.fc = nn.Linear(m.fc.in_features, 2)
#     m.load_state_dict(torch.load(weights_path, map_location=device))
#     m.to(device)
#     m.eval()
#     return m

# model_sand = load_model("resnet_wdpt.pth")
# model_topsoil = load_model("resnet_wdpt_topsoil.pth")

# # ----- TRANSFORMS -----
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# ])

# # ----- UTILS -----
# def delete_folder(folder):
#     if os.path.exists(folder):
#         shutil.rmtree(folder)
#     os.makedirs(folder)

# def extract_last_frame(video_path, roi_coords):
#     x1,y1,x2,y2 = roi_coords
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         roi = frame[y1:y2, x1:x2].copy()
#         frames.append(roi)
#     cap.release()
#     last_frame = Image.fromarray(frames[-1])
#     return last_frame

# def classify_frame(image):
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         pred_sand = model_sand(image).argmax(dim=1).item()
#         pred_topsoil = model_topsoil(image).argmax(dim=1).item()
#     return min(pred_sand, pred_topsoil)

# def analyze_wdpt(video_path, roi_coords):
#     x1,y1,x2,y2 = roi_coords
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     backSub = cv2.createBackgroundSubtractorMOG2(history=300,varThreshold=16,detectShadows=False)
    
#     start_time = None
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret: break
#         roi = frame[y1:y2, x1:x2]
#         fg_mask = backSub.apply(roi)
#         kernel = np.ones((5,5),np.uint8)
#         fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
#         activity = np.count_nonzero(fg_mask)
#         frames.append(roi)
#         if start_time is None and activity>1000:
#             start_time = len(frames)/fps
#     cap.release()
#     if start_time is None:
#         start_time = 0
    
#     # reverse detection
#     final_ref = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
#     final_ref = cv2.GaussianBlur(final_ref,(5,5),0)
#     end_frame_idx = None
#     for i in range(len(frames)-1, -1, -1):
#         cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
#         cur_gray = cv2.GaussianBlur(cur_gray,(5,5),0)
#         diff = cv2.absdiff(cur_gray, final_ref)
#         _, thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)
#         thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
#         score = np.count_nonzero(thresh)
#         if score>50:
#             end_frame_idx = i
#             break
#     end_time = end_frame_idx/fps if end_frame_idx else 0
#     wdpt = end_time-start_time
#     return wdpt

# # ----- MAC → PI SOCKET -----
# PI_IP = "192.168.1.100"  # set Pi IP
# PI_PORT = 5005
# def send_pi_command(cmd):
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.connect((PI_IP, PI_PORT))
#     s.sendall(cmd.encode())
#     s.close()

# # ----- ARDUINO SETUP -----
# ARDUINO_PORT = "/dev/tty.usbmodem14101"  # check your port
# arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
# time.sleep(2)  # allow Arduino to reset

# def run_pump(duration=2):
#     arduino.write(b'ON\n')
#     time.sleep(duration)
#     arduino.write(b'OFF\n')

# # ----- MAIN SEQUENCE -----
# def main():
#     roi_coords = (684,185,984,377)
#     video_file = "capture.mov"
    
#     # 1. Forward mobility demo
#     print("Sending forward command to Pi...")
#     send_pi_command("forward")
    
#     # 2. Record live video (2 min max)
#     print("Recording video...")
#     cap = cv2.VideoCapture(0)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(video_file, fourcc, 30, (640,480))
#     start = time.time()
#     while time.time()-start < 120:
#         ret, frame = cap.read()
#         if not ret: break
#         out.write(frame)
#     cap.release()
#     out.release()
#     print("Video recorded.")
    
#     # 3. Start pump
#     print("Running pump...")
#     run_pump(duration=10)
    
#     # 4. CNN detection
#     last_frame = extract_last_frame(video_file, roi_coords)
#     pred = classify_frame(last_frame)
#     if pred==0:
#         print("Absorption not detected. Skipping WDPT.")
#         send_pi_command("no_till")
#         return
    
#     # 5. WDPT detection
#     wdpt = analyze_wdpt(video_file, roi_coords)
#     print(f"WDPT: {wdpt:.2f}s")
    
#     # 6. Send tillage commands
#     if wdpt<5:
#         cmd="no_till"
#     elif wdpt<=60:
#         cmd="shallow_till"
#     else:
#         cmd="deep_till"
#     print(f"Sending command to Pi: {cmd}")
#     send_pi_command(cmd)

# if __name__=="__main__":
#     main()


import cv2
import time
import socket
import serial
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------- SETTINGS ----------------
PI_IP = "192.168.2.2"     # Pi static IP
PI_PORT = 5005            # Port for Pi socket
ARDUINO_PORT = "/dev/cu.usbmodem1201"  # Your Arduino Uno
VIDEO_DURATION = 120       # 2 min cutoff in seconds

# ---------------- CNN SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(weights_path, map_location=device))
    m.to(device)
    m.eval()
    return m

model_sand = load_model("resnet_wdpt.pth")
model_topsoil = load_model("resnet_wdpt_topsoil.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- ARDUINO ----------------
arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
time.sleep(2)  # allow Arduino reset

def run_pump(duration=5):
    arduino.write(b'ON\n')
    print("Pump ON")
    time.sleep(duration)
    arduino.write(b'OFF\n')
    print("Pump OFF")

# ---------------- PI SOCKET ----------------
pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
pi_sock.connect((PI_IP, PI_PORT))

def send_pi_command(cmd):
    pi_sock.sendall(cmd.encode())
    print(f"Sent to Pi: {cmd}")

# ---------------- CAMERA CAPTURE ----------------
cap = cv2.VideoCapture(0)  # Mac camera
start_time = time.time()
frames = []

print("Starting 2-minute recording...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    cv2.imshow("Live Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time >= VIDEO_DURATION:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {len(frames)} frames for analysis.")

# ---------------- WDPT DETECTION ----------------
def classify_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_sand = model_sand(img).argmax(dim=1).item()
        pred_topsoil = model_topsoil(img).argmax(dim=1).item()
    final_pred = min(pred_sand, pred_topsoil)
    return final_pred

# For simplicity, check the **last frame**
wdpt_pred = classify_frame(frames[-1])
print(f"ML-based absorption check: {wdpt_pred} -> {'absorbed' if wdpt_pred==1 else 'not absorbed'}")

# ---------------- SIMPLE WDPT TIMING ----------------
# Using frame differences as a proxy for absorption time
def calculate_wdpt(frames):
    start_frame = frames[0]
    for idx, f in enumerate(frames):
        diff = cv2.absdiff(cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        score = np.count_nonzero(diff)
        if score > 1000:  # threshold for droplet landing
            start_idx = idx
            break
    end_idx = len(frames) - 1  # assume last frame absorption complete
    wdpt_time = (end_idx - start_idx)/30  # approximate FPS ~30
    return wdpt_time

if wdpt_pred == 1:
    wdpt_time = calculate_wdpt(frames)
    print(f"Estimated WDPT: {wdpt_time:.2f}s")

    # ---------------- TILLAGE DECISION ----------------
    if wdpt_time < 5:
        send_pi_command("NO_TILL")
    elif wdpt_time < 60:
        send_pi_command("SHALLOW_TILL")
    else:
        send_pi_command("DEEP_TILL")
else:
    print("WDPT not detected. Skipping tillage.")
    send_pi_command("NO_TILL")

# ---------------- RUN PUMP ----------------
run_pump(duration=5)

# ---------------- STOP ----------------
pi_sock.close()
arduino.close()
print("Program complete.")
