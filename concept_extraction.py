import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# --- Paths ---
model_path = '/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/pose_landmarker_lite.task'
input_video_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/videos/dancing ballet/0owoOHazQvU_534_544.mp4"
output_video_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/videos/dancing ballet/fullbodypose/0owoOHazQvU_534_544_fullbodypose.mp4"

# --- MediaPipe setup ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# --- Open video ---
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Setup output video ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- First pass: collect all foot landmarks to compute fixed square ---
all_x = []
all_y = []
frames_list = []

with PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, int(frame_idx / fps * 1000))
        frames_list.append(frame.copy())  # store frame for second pass

        if result.pose_landmarks:
            # Assuming single person
            landmarks = result.pose_landmarks[0]

            l_ankle = landmarks[27]
            l_foot = landmarks[31]
            r_ankle = landmarks[28]
            r_foot = landmarks[32]

            all_x.extend([l_ankle.x, l_foot.x, r_ankle.x, r_foot.x])
            all_y.extend([l_ankle.y, l_foot.y, r_ankle.y, r_foot.y])

        frame_idx += 1

cap.release()

# --- Compute fixed square bounding box for entire video ---
x_min = int(min(all_x) * width)
x_max = int(max(all_x) * width)
y_min = int(min(all_y) * height)
y_max = int(max(all_y) * height)

box_w = x_max - x_min
box_h = y_max - y_min
size = max(box_w, box_h)
pad = int(0.1 * size)
size += 2*pad

center_x = (x_min + x_max)//2
center_y = (y_min + y_max)//2

x1 = max(0, center_x - size//2)
y1 = max(0, center_y - size//2)
x2 = min(width, center_x + size//2)
y2 = min(height, center_y + size//2)

print(f"Fixed feet box: ({x1}, {y1}) -> ({x2}, {y2})")

# --- Second pass: annotate video + prepare frames for TCAV ---
feet_frames = []
with PoseLandmarker.create_from_options(options) as landmarker:
    for idx, frame in enumerate(frames_list):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, int(idx / fps * 1000))

        # Draw skeleton
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for lm in landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw fixed feet box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Write annotated frame
        out.write(frame)

        # Crop for TCAV
        feet_crop = frame[y1:y2, x1:x2]
        feet_crop_tensor = torch.tensor(feet_crop).permute(2,0,1).unsqueeze(0).float() / 255.0
        feet_crop_tensor = F.interpolate(feet_crop_tensor, size=(112,112), mode='bilinear', align_corners=False)
        feet_frames.append(feet_crop_tensor)

out.release()
cv2.destroyAllWindows()

# --- Stack frames for 3D CNN: [1, C, T, H, W] ---
clip_tensor = torch.cat(feet_frames, dim=0)   # [T, C, H, W]
clip_tensor = clip_tensor.unsqueeze(0)        # add batch: [1, T, C, H, W]
clip_tensor = clip_tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

print("Clip tensor ready for R3D-18 / TCAV:", clip_tensor.shape)
