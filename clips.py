import os
import torch
from torchvision.io import read_video
import torch.nn.functional as F
import sys
# Add yolov7 folder to Python path
sys.path.append(os.path.join(os.getcwd(), "yolov7"))

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression
from yolov7.utils.datasets import letterbox
import itertools
import math
import cv2

# -----------------------------
# Settings
# -----------------------------
VIDEO_ROOT = "videos/salsa dancing"    # downloaded clips
OUTPUT_ROOT = "concept_tcav_videos"    # save concept & random clips
CLIP_LEN = 16
DIST_THRESHOLD = 150                   # pixels
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 10                               # fps for output video clips

os.makedirs(os.path.join(OUTPUT_ROOT, "concept"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "random"), exist_ok=True)

# -----------------------------
# Load YOLOv7
# -----------------------------
MODEL_PATH = "yolov7/yolov7.pt"
model = attempt_load(MODEL_PATH, map_location=DEVICE)
model.eval()

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_frame(frame, img_size=640):
    """
    Prepares a single frame for YOLOv7 detection.
    - frame: H x W x C (RGB) numpy array from read_video
    - img_size: size for letterbox resizing
    Returns: 1 x 3 x H x W tensor on the correct device
    """
    # Letterbox resize (maintains aspect ratio, pads)
    img = letterbox(frame, new_shape=img_size)[0]

    # Make a contiguous copy to avoid negative strides
    img = img.copy()

    # Channels-first
    img = img.transpose(2, 0, 1)

    # Convert to float tensor, scale 0-1
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(DEVICE)  # add batch dimension
    return img


def detect_persons(frame):
    img = preprocess_frame(frame)
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]
    boxes = []
    if pred is not None:
        for *xyxy, conf, cls in pred.cpu().numpy():
            if int(cls)==0:
                boxes.append(xyxy)
    return boxes

def avg_person_distance(boxes):
    if len(boxes)<2:
        return float("inf")
    centers = [((x1+x2)/2,(y1+y2)/2) for x1,y1,x2,y2 in boxes]
    distances = [math.dist(c1,c2) for c1,c2 in itertools.combinations(centers,2)]
    return sum(distances)/len(distances)

def crop_around_boxes(frame, boxes, target_size=(112,112)):
    if len(boxes) == 0:
        return None  # skip frame if no boxes detected

    x1s, y1s, x2s, y2s = zip(*boxes)
    x1, y1 = max(int(min(x1s)-10), 0), max(int(min(y1s)-10), 0)
    x2, y2 = min(int(max(x2s)+10), frame.shape[1]), min(int(max(y2s)+10), frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None  # skip invalid crop

    cropped = frame[y1:y2, x1:x2, :]
    if target_size is not None:
        cropped = cv2.resize(cropped, target_size)
    return cropped



def preprocess_clip(frames):
    """Prepare for TCAV [1,C,T,H,W]"""
    frames = torch.stack(frames).permute(0,3,1,2).float()/255.0
    T = frames.shape[0]
    if T<CLIP_LEN:
        return None
    idx = torch.linspace(0,T-1,CLIP_LEN).long()
    frames = frames[idx]

    _,C,H,W = frames.shape
    short_edge = 128
    if H<W:
        new_h = short_edge
        new_w = int(W*short_edge/H)
    else:
        new_w = short_edge
        new_h = int(H*short_edge/W)
    frames = F.interpolate(frames, size=(new_h,new_w), mode="bilinear", align_corners=False)

    top = (new_h-112)//2
    left = (new_w-112)//2
    frames = frames[:,:,top:top+112,left:left+112]

    mean = torch.tensor([0.43216,0.394666,0.37645]).view(1,3,1,1)
    std = torch.tensor([0.22803,0.22145,0.216989]).view(1,3,1,1)
    frames = (frames-mean)/std
    frames = frames.unsqueeze(0).permute(0,2,1,3,4)
    return frames

def save_clip_as_video(frames_list, out_path, fps=10):
    if len(frames_list)==0:
        return
    h,w,_ = frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path,fourcc,fps,(w,h))
    for f in frames_list:
        if isinstance(f, torch.Tensor):
            f = f.numpy()
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(f_bgr)
    out.release()

def make_subclips(frames_list, out_dir, prefix, save_video=True, save_tensor=True):
    for i in range(0,len(frames_list)-CLIP_LEN+1, CLIP_LEN):
        clip_frames = frames_list[i:i+CLIP_LEN]
        if save_video:
            save_clip_as_video(clip_frames, os.path.join(out_dir,f"{prefix}_{i}.mp4"), fps=FPS)
        if save_tensor:
            processed = preprocess_clip(clip_frames)
            if processed is not None:
                torch.save(processed, os.path.join(out_dir,f"{prefix}_{i}.pt"))

# -----------------------------
# Process all videos
# -----------------------------
for file in os.listdir(VIDEO_ROOT):
    video_path = os.path.join(VIDEO_ROOT,file)
    print(f"Processing {file}...")
    frames, _, _ = read_video(video_path, pts_unit="sec")
    frames_concept, frames_random = [], []

    for f in frames:
        f_np = f.numpy()
        boxes = detect_persons(f_np)
        avg_dist = avg_person_distance(boxes)
        cropped = crop_around_boxes(f_np, boxes, target_size=(112,112))
        if cropped is not None:
            cropped_tensor = torch.from_numpy(cropped)
            if avg_dist < DIST_THRESHOLD:
                frames_concept.append(cropped_tensor)
            else:
                frames_random.append(cropped_tensor)


    make_subclips(frames_concept, os.path.join(OUTPUT_ROOT,"concept"), file.split(".")[0])
    make_subclips(frames_random, os.path.join(OUTPUT_ROOT,"random"), file.split(".")[0])

print("All concept and random clips generated! You can watch them and/or feed to TCAV.")
