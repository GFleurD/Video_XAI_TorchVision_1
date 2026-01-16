import cv2
import mediapipe as mp
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
model_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/pose_landmarker_lite.task"
input_video_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/videos/salsa dancing/-_ScQW-_JMQ_8_18.mp4"

clips_dir = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/videos/salsa dancing/clips"
os.makedirs(clips_dir, exist_ok=True)

video_name = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = os.path.join(clips_dir, f"{video_name}_clip.mp4")

# --------------------------------------------------
# MediaPipe setup
# --------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# --------------------------------------------------
# Open video
# --------------------------------------------------
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
all_x, all_y = [], []

# --------------------------------------------------
# First pass: compute fixed feet bounding box
# --------------------------------------------------
with PoseLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(
            mp_image, int(frame_idx / fps * 1000)
        )

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]

            for idx in [27, 28, 31, 32]:  # ankles + feet
                all_x.append(lm[idx].x)
                all_y.append(lm[idx].y)

        frame_idx += 1

cap.release()

if not all_x:
    raise RuntimeError("No pose landmarks detected in video.")

# --------------------------------------------------
# Compute fixed square crop (yellow box)
# --------------------------------------------------
x_min = int(min(all_x) * width)
x_max = int(max(all_x) * width)
y_min = int(min(all_y) * height)
y_max = int(max(all_y) * height)

box_w = x_max - x_min
box_h = y_max - y_min
size = max(box_w, box_h)

pad = int(0.1 * size)
size += 2 * pad

cx = (x_min + x_max) // 2
cy = (y_min + y_max) // 2

x1 = max(0, cx - size // 2)
y1 = max(0, cy - size // 2)
x2 = min(width, cx + size // 2)
y2 = min(height, cy + size // 2)

clip_w = x2 - x1
clip_h = y2 - y1

print(f"Fixed clip box: ({x1}, {y1}) → ({x2}, {y2})")

# --------------------------------------------------
# Second pass: write clipped video
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (clip_w, clip_h))

for frame in frames:
    crop = frame[y1:y2, x1:x2]
    out.write(crop)

out.release()
cv2.destroyAllWindows()

print(f"Clipped video saved to:\n{output_video_path}")
