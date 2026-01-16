import os
import json
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from torchvision.io import read_video
from sklearn.linear_model import LogisticRegression
from torchvision.models.video import r3d_18

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CONCEPT_DIR = "data/concept"
RANDOM_DIR  = "data/random"
CLASS_JSON  = "data/kinetics_classnames1.json"

TARGET_LABEL = "dancing ballet"
BOTTLENECK_LAYER = "layer4.1.conv2"
NUM_FRAMES = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load class mapping
# --------------------------------------------------
with open(CLASS_JSON, "r") as f:
    id_to_class = json.load(f)

def label_to_index(label):
    for idx, name in id_to_class.items():
        if name == label:
            return int(idx)
    raise ValueError(f"Label '{label}' not found.")

TARGET_CLASS_IDX = label_to_index(TARGET_LABEL)

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = r3d_18(weights="DEFAULT").to(DEVICE).eval()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_video(video_path):
    frames, _, _ = read_video(video_path, pts_unit="sec")
    frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

    # temporal sampling
    T = frames.shape[0]
    idx = torch.linspace(0, T - 1, NUM_FRAMES).long()
    frames = frames[idx]

    # resize short edge to 128
    _, C, H, W = frames.shape
    short_edge = 128

    if H < W:
        new_h = short_edge
        new_w = int(W * short_edge / H)
    else:
        new_w = short_edge
        new_h = int(H * short_edge / W)

    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False
    )

    # center crop 112x112
    top = (new_h - 112) // 2
    left = (new_w - 112) // 2
    frames = frames[:, :, top:top+112, left:left+112]

    # normalize (Kinetics)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
    std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    # [B, C, T, H, W]
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return frames.to(DEVICE)

# --------------------------------------------------
# TCAV Wrapper
# --------------------------------------------------
class TCAVWrapper:
    def __init__(self, model, bottleneck_layer):
        self.model = model
        self.layer_name = bottleneck_layer
        self.activations = {}

        layer = dict(self.model.named_modules())[bottleneck_layer]
        layer.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.activations[self.layer_name] = output
        output.retain_grad()

    def get_acts_and_grads(self, x, target_class):
        self.model.zero_grad()
        preds = self.model(x)
        preds[0, target_class].backward()
        acts = self.activations[self.layer_name]
        grads = acts.grad
        return acts.detach(), grads.detach()

wrapper = TCAVWrapper(model, BOTTLENECK_LAYER)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
def load_dataset(dir_path):
    videos = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".mp4"):
            path = os.path.join(dir_path, fname)
            videos.append(preprocess_video(path))
    return videos

concept_videos = load_dataset(CONCEPT_DIR)
random_videos  = load_dataset(RANDOM_DIR)

assert len(concept_videos) > 1, "Need more than one concept example"
assert len(random_videos)  > 1, "Need more than one random example"

# --------------------------------------------------
# Collect activations
# --------------------------------------------------
def flatten_acts(acts):
    return acts.reshape(acts.shape[0], -1)

def collect_activations(videos):
    all_acts = []
    for v in videos:
        acts, _ = wrapper.get_acts_and_grads(v, TARGET_CLASS_IDX)
        all_acts.append(flatten_acts(acts))
    return torch.cat(all_acts, dim=0)

concept_acts = collect_activations(concept_videos)
random_acts  = collect_activations(random_videos)

# --------------------------------------------------
# Train CAV
# --------------------------------------------------
X = torch.cat([concept_acts, random_acts], dim=0).cpu().numpy()
y = np.array(
    [1] * concept_acts.shape[0] +
    [0] * random_acts.shape[0]
)

cav_model = LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    class_weight="balanced"
)

cav_model.fit(X, y)
cav = cav_model.coef_.flatten()

# --------------------------------------------------
# Compute TCAV score (on concept set)
# --------------------------------------------------
def tcav_score(videos):
    scores = []
    for v in videos:
        acts, grads = wrapper.get_acts_and_grads(v, TARGET_CLASS_IDX)
        acts_f = flatten_acts(acts).cpu().numpy()
        grads_f = grads.reshape(grads.shape[0], -1).cpu().numpy()
        directional_deriv = np.dot(grads_f, cav)
        scores.append(np.mean(directional_deriv > 0))
    return np.mean(scores)

score = tcav_score(concept_videos)

print(f"\nTCAV score for '{TARGET_LABEL}': {score:.4f}")
print(f"Concept clips: {len(concept_videos)}")
print(f"Random clips:  {len(random_videos)}")
