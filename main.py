import cv2
import time
import socket
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import threading

# ---------------- SETTINGS ----------------
PI_IP = "192.168.2.2"     # Pi static IP
PI_PORT = 5005            # Port for Pi socket
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

# ---------------- PI CONNECTION ----------------
pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
pi_sock.connect((PI_IP, PI_PORT))
print("Connected to Pi motor server!")

def send_pi_command(command):
    """Send a simple string command to the Pi."""
    pi_sock.sendall(command.encode())
    print(f"Sent to Pi: {command}")

# ---------------- CAMERA CAPTURE ----------------
cap = cv2.VideoCapture(0)  # Mac camera
start_time = time.time()
frames = []

print("Starting 2-minute recording...")

# Start pump immediately
threading.Thread(target=lambda: send_pi_command("PUMP_ON")).start()

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

# Check last frame for absorption
wdpt_pred = classify_frame(frames[-1])
print(f"ML-based absorption check: {wdpt_pred} -> {'absorbed' if wdpt_pred==1 else 'not absorbed'}")

# ---------------- SIMPLE WDPT TIMING ----------------
def calculate_wdpt(frames):
    start_frame = frames[0]
    start_idx = 0
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

# ---------------- TILLAGE DECISION ----------------
if wdpt_pred == 1:
    wdpt_time = calculate_wdpt(frames)
    print(f"Estimated WDPT: {wdpt_time:.2f}s")

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
send_pi_command("PUMP_ON")

# ---------------- CLEANUP ----------------
pi_sock.close()
print("Program complete.")
