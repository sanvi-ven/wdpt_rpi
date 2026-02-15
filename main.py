# # import cv2
# # import time
# # import socket
# # import torch
# # import numpy as np
# # from PIL import Image
# # from torchvision import transforms
# # from torchvision.models import resnet18, ResNet18_Weights
# # import threading

# # # ---------------- SETTINGS ----------------
# # PI_IP = "192.168.2.2"     # Pi static IP
# # PI_PORT = 5005            # Port for Pi socket
# # VIDEO_DURATION = 120       # 2 min cutoff in seconds

# # # ---------------- CNN SETUP ----------------
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # def load_model(weights_path):
# #     m = resnet18(weights=ResNet18_Weights.DEFAULT)
# #     m.fc = torch.nn.Linear(m.fc.in_features, 2)
# #     m.load_state_dict(torch.load(weights_path, map_location=device))
# #     m.to(device)
# #     m.eval()
# #     return m

# # model_sand = load_model("resnet_wdpt.pth")
# # model_topsoil = load_model("resnet_wdpt_topsoil.pth")

# # transform = transforms.Compose([
# #     transforms.Resize((224, 224)),
# #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
# #     transforms.ToTensor(),
# #     transforms.Normalize(
# #         mean=[0.485, 0.456, 0.406],
# #         std=[0.229, 0.224, 0.225]
# #     )
# # ])

# # # ---------------- PI CONNECTION ----------------
# # pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # pi_sock.connect((PI_IP, PI_PORT))
# # print("Connected to Pi motor server!")

# # def send_pi_command(command):
# #     """Send a simple string command to the Pi."""
# #     pi_sock.sendall(command.encode())
# #     print(f"Sent to Pi: {command}")

# # # ---------------- CAMERA CAPTURE ----------------
# # cap = cv2.VideoCapture(0)  # Mac camera
# # start_time = time.time()
# # frames = []

# # print("Starting 2-minute recording...")

# # # Start pump immediately
# # threading.Thread(target=lambda: send_pi_command("PUMP_ON")).start()

# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #     frames.append(frame)
# #     cv2.imshow("Live Capture", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #     if time.time() - start_time >= VIDEO_DURATION:
# #         break

# # cap.release()
# # cv2.destroyAllWindows()
# # print(f"Captured {len(frames)} frames for analysis.")

# # # ---------------- WDPT DETECTION ----------------
# # def classify_frame(frame):
# #     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
# #     img = transform(img).unsqueeze(0).to(device)
# #     with torch.no_grad():
# #         pred_sand = model_sand(img).argmax(dim=1).item()
# #         pred_topsoil = model_topsoil(img).argmax(dim=1).item()
# #     final_pred = min(pred_sand, pred_topsoil)
# #     return final_pred

# # # Check last frame for absorption
# # wdpt_pred = classify_frame(frames[-1])
# # print(f"ML-based absorption check: {wdpt_pred} -> {'absorbed' if wdpt_pred==1 else 'not absorbed'}")

# # # ---------------- SIMPLE WDPT TIMING ----------------
# # def calculate_wdpt(frames):
# #     start_frame = frames[0]
# #     start_idx = 0
# #     for idx, f in enumerate(frames):
# #         diff = cv2.absdiff(cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY),
# #                            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
# #         score = np.count_nonzero(diff)
# #         if score > 1000:  # threshold for droplet landing
# #             start_idx = idx
# #             break
# #     end_idx = len(frames) - 1  # assume last frame absorption complete
# #     wdpt_time = (end_idx - start_idx)/30  # approximate FPS ~30
# #     return wdpt_time

# # # ---------------- TILLAGE DECISION ----------------
# # if wdpt_pred == 1:
# #     wdpt_time = calculate_wdpt(frames)
# #     print(f"Estimated WDPT: {wdpt_time:.2f}s")

# #     if wdpt_time < 5:
# #         send_pi_command("NO_TILL")
# #     elif wdpt_time < 60:
# #         send_pi_command("SHALLOW_TILL")
# #     else:
# #         send_pi_command("DEEP_TILL")
# # else:
# #     print("WDPT not detected. Skipping tillage.")
# #     send_pi_command("NO_TILL")


# # # ---------------- CLEANUP ----------------
# # pi_sock.close()
# # print("Program complete.")


# import cv2
# import time
# import socket
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torchvision.models import resnet18, ResNet18_Weights

# # ---------------- SETTINGS ----------------
# PI_IP = "192.168.2.2"
# PI_PORT = 5005
# VIDEO_DURATION = 120  # seconds

# # ---------------- CNN SETUP ----------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_model(weights_path):
#     m = resnet18(weights=ResNet18_Weights.DEFAULT)
#     m.fc = torch.nn.Linear(m.fc.in_features, 2)
#     m.load_state_dict(torch.load(weights_path, map_location=device))
#     m.to(device)
#     m.eval()
#     return m

# model_sand = load_model("resnet_wdpt.pth")
# model_topsoil = load_model("resnet_wdpt_topsoil.pth")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# ])

# # ---------------- PI CONNECTION ----------------
# pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# pi_sock.connect((PI_IP, PI_PORT))
# print("Connected to Pi motor server!")

# def send_pi_command(command):
#     pi_sock.sendall(command.encode())
#     print(f"Sent to Pi: {command}")

# # ---------------- STEP 1: MOVE FORWARD ----------------
# print("Moving forward...")
# send_pi_command("FORWARD")
# time.sleep(3)  # wait for forward motion to complete (match motor_server sleep duration)

# # ---------------- STEP 2: RUN PUMP ----------------
# print("Starting pump...")
# send_pi_command("PUMP_ON")

# # ---------------- STEP 3: CAMERA CAPTURE ----------------
# cap = cv2.VideoCapture(0)
# start_time = time.time()
# frames = []

# print("Starting 2-minute recording...")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frames.append(frame)
#     cv2.imshow("Live Capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     if time.time() - start_time >= VIDEO_DURATION:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print(f"Captured {len(frames)} frames for analysis.")

# # ---------------- WDPT DETECTION ----------------
# def classify_frame(frame):
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
#     img = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         pred_sand = model_sand(img).argmax(dim=1).item()
#         pred_topsoil = model_topsoil(img).argmax(dim=1).item()
#     return min(pred_sand, pred_topsoil)

# wdpt_pred = classify_frame(frames[-1])
# print(f"ML-based absorption check: {wdpt_pred} -> {'absorbed' if wdpt_pred==1 else 'not absorbed'}")

# # ---------------- SIMPLE WDPT TIMING ----------------
# def calculate_wdpt(frames):
#     start_frame = frames[0]
#     start_idx = 0
#     for idx, f in enumerate(frames):
#         diff = cv2.absdiff(cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY),
#                            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
#         if np.count_nonzero(diff) > 1000:
#             start_idx = idx
#             break
#     wdpt_time = (len(frames) - start_idx) / 30  # assume 30 FPS
#     return wdpt_time

# # ---------------- TILLAGE DECISION ----------------
# if wdpt_pred == 1:
#     wdpt_time = calculate_wdpt(frames)
#     print(f"Estimated WDPT: {wdpt_time:.2f}s")
#     if wdpt_time < 5:
#         send_pi_command("NO_TILL")
#     elif wdpt_time < 60:
#         send_pi_command("SHALLOW_TILL")
#     else:
#         send_pi_command("DEEP_TILL")
# else:
#     print("WDPT not detected. Skipping tillage.")
#     send_pi_command("NO_TILL")

# # ---------------- CLEANUP ----------------
# pi_sock.close()
# print("Program complete.")


import cv2
import time
import socket
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import shutil
import torch.nn as nn
import threading

# ---------------- SETTINGS ----------------
PI_IP = "192.168.2.2"
PI_PORT = 5005
VIDEO_DURATION = 30  # seconds
ROI_COORDS = (735, 161, 1259, 400)  # x1, y1, x2, y2
FPS_APPROX = 30  # approximate capture FPS
PUMP_DELAY = 5.0  # seconds to wait after recording starts before turning pump on


# ---------------- PI CONNECTION ----------------
pi_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
pi_sock.connect((PI_IP, PI_PORT))
print("Connected to Pi motor server!")

def send_pi_command(command):
    pi_sock.sendall(command.encode())
    print(f"Sent to Pi: {command}")

# ---------------- FOLDER HELPERS ----------------
def delete_folder(folder):
    folder_path = os.path.abspath(folder)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)

# ---------------- CAMERA + FRAME EXTRACTION ----------------
def capture_frames(output_folder="frame_inspection", fps_out=5):
    delete_folder(output_folder)
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    frames = []
    # Read camera FPS; some backends return 0.0 so fallback to FPS_APPROX.
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = FPS_APPROX
    # Ensure at least every 1st frame is considered (avoid frame_interval == 0)
    frame_interval = max(1, int(round(video_fps / fps_out)))
    count = 0
    saved = 0

    # Background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)

    print("Starting live capture and saving frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1

        # Crop ROI
        x1, y1, x2, y2 = ROI_COORDS
        roi = frame[y1:y2, x1:x2]
        kernel = np.ones((5,5), np.uint8)
        # Background subtraction + morph
        fg_mask = backSub.apply(roi)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Save and append ONLY every Nth frame so `frames` == saved frames
        if count % frame_interval == 0:
            frame_name = f"frame_{saved:04d}.jpg"
            saved_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(saved_path, roi)
            saved += 1
            frames.append(roi)

        # Show live
        cv2.imshow("Live ROI Capture", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time.time() - start_time >= VIDEO_DURATION:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {len(frames)} frames. Saved {saved} ROI frames.")
    return frames, output_folder

# # ---------------- CNN CLASSIFICATION ----------------
# def classify_last_frame(folder_path):
#     files = sorted(os.listdir(folder_path))
#     last_frame_path = os.path.join(folder_path, files[-1])
#     image = Image.open(last_frame_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         pred_sand = model_sand(image).argmax(dim=1).item()
#         pred_topsoil = model_topsoil(image).argmax(dim=1).item()

#     final_pred = min(pred_sand, pred_topsoil)
#     print(f"{last_frame_path} -> sand:{pred_sand}, topsoil:{pred_topsoil} -> final:{final_pred}")
#     return final_pred


# ---------------- BACKWARD DETECTION ----------------
def backward_wdpt(frames, fps=FPS_APPROX, output_folder="reverse_analysis"):
    delete_folder(output_folder)
    total = len(frames)
    final_reference = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    final_reference = cv2.GaussianBlur(final_reference, (5,5), 0)

    detected_frame_idx = None
    for i in range(total-1, -1, -1):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        diff = cv2.absdiff(gray, final_reference)
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        diff_score = np.count_nonzero(thresh)
        if diff_score > 215:
            detected_frame_idx = i
            cv2.imwrite(os.path.join(output_folder, "detected_end_point.jpg"), frames[i])
            print(f"Backward detection end at frame {i} ({i/fps:.2f}s)")
            return i/fps
    return None

# ---------------- WDPT START DETECTION ----------------
def forward_wdpt(frames, fps=FPS_APPROX, collect_activity=False):
    """Detect start time (when droplet lands) in the provided ROI frames.

    Args:
        frames (list): list of ROI frames (BGR numpy arrays).
        fps (float): frames-per-second assumed for timing.
        collect_activity (bool): if True, also collect and return the per-frame activity log.

    Returns:
        float or (float, list): start_time (seconds) or (start_time, activity_log) if collect_activity True.
        If no start detected or no frames provided, returns None (or (None, activity_log)).
    """
    # If no frames were provided, nothing to detect
    if not frames:
        print("forward_wdpt: no frames to analyze")
        return (None, []) if collect_activity else None

    # Use a fresh background subtractor (like the integration script)
    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
    start_time = None
    kernel = np.ones((5,5), np.uint8)
    activity_log = []

    for idx, frame in enumerate(frames):
        # Ensure frame is BGR color image (bgSub expects color frames);
        # if frame is grayscale, convert to BGR so apply() behaves consistently.
        if frame is None:
            activity_log.append(0)
            continue
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            frame_proc = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_proc = frame

        fg_mask = backSub.apply(frame_proc)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        activity_score = int(np.count_nonzero(fg_mask))
        activity_log.append(activity_score)

        # same threshold as integration script
        if start_time is None and activity_score > 1000:
            start_time = idx / fps + PUMP_DELAY  # adjust start time to account for pump delay
            print(f"Start detected at frame {idx} ({start_time:.2f}s)")
            if collect_activity:
                return start_time, activity_log
            return start_time

    # No start detected
    if collect_activity:
        return None, activity_log
    return None

# ---------------- MAIN PROGRAM ----------------
# 1️⃣ Move forward first
send_pi_command("SHALLOW_TILL")
time.sleep(3)

# 2️⃣ Start pump a short time AFTER recording begins (non-blocking)
# Schedule the pump to start once capture is underway so the recorder has a warm-up frame
threading.Timer(PUMP_DELAY, lambda: send_pi_command("PUMP_ON")).start()

# 3️⃣ Capture live frames with ROI + BG Subtraction
frames, folder = capture_frames(fps_out=5)

# 4️⃣ CNN classification on last frame

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
    img_path = "/Users/sanviadmin/Documents/GitHub/wdpt_rpi/frame_inspection/frame_0855.jpg"
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_sand = model_sand(inp).argmax(dim=1).item()
        pred_topsoil = model_topsoil(inp).argmax(dim=1).item()
    final_pred = min(pred_sand, pred_topsoil)
    print(f"Prediction: {img_path} -> sand:{pred_sand}, topsoil:{pred_topsoil} -> final:{final_pred}")
print(f"ML-based absorption check: {final_pred} -> {'absorbed' if final_pred==1 else 'not absorbed'}")

# Only do WDPT timing if absorption detected
if final_pred == 1:
    start = forward_wdpt(frames)
    end = backward_wdpt(frames)
    if start is not None and end is not None:
        wdpt_time = end - start
        print(f"WDPT estimated: {wdpt_time:.2f}s")

        # Automatic tillage decision
        if wdpt_time < 5:
            send_pi_command("NO_TILL")
            print("Tillage decision: NO_TILL")
        elif wdpt_time < 60:
            send_pi_command("SHALLOW_TILL")
            print("Tillage decision: SHALLOW_TILL")
elif final_pred == 0:
    send_pi_command("DEEP_TILL")
    print("Tillage decision: DEEP_TILL")
else:
    print("Absorption not detected; skipping WDPT timing and tillage decision")

# 7️⃣ Cleanup
pi_sock.close()
print("Program complete.")
