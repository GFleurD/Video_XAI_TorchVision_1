import torch
import torchvision
from torchvision.io import read_video
import json 

import json

# Load your current JSON (class name → index)
with open("kinetics_classnames.json", "r") as f:
    old_mapping = json.load(f)

# Invert it: index (as string) → class name (no extra quotes)
new_mapping = {str(v): k.replace('"', '') for k, v in old_mapping.items()}

# Save it as kinetics_classnames.json
with open("kinetics_classnames1.json", "w") as f:
    json.dump(new_mapping, f, indent=2)

print("Done! You now have a correct index → class name JSON.")


# --- Load class‑id → name mapping ---
# You need a file kinetics_classnames.json in same folder (you can download from PyTorchVideo tutorial)
with open("kinetics_classnames1.json", "r") as f:
    id_to_class = json.load(f)

# --- Load model ---
model = torchvision.models.video.r3d_18(weights="DEFAULT")
model = model.eval()

# --- Load video clip ---
video_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/Baking_Christmas_Cookies_Clip.mp4"
frames, _, _ = read_video(video_path, pts_unit='sec')

# --- Preprocess ---
frames = frames.permute(0, 3, 1, 2)            # [T, C, H, W]
frames = frames[:16].float() / 255.0          # first 16 frames, normalise
frames = frames.unsqueeze(0)                  # [1, T, C, H, W] as neural networks always expect batches even of size 1
frames = frames.permute(0, 2, 1, 3, 4)         # [B, C, T, H, W]

# --- Hook for intermediate activations ---
activations = {}
def hook_fn(module, input, output):
    activations['layer4_block'] = output

model.layer4[1].conv2.register_forward_hook(hook_fn)

# --- Forward pass + predictions ---
with torch.no_grad():
    preds = model(frames)

probs = torch.softmax(preds, dim=1)[0]
top5 = torch.topk(probs, 5)

print("Top‑5 predictions:")
for idx, score in zip(top5.indices, top5.values):
    label = id_to_class.get(str(int(idx)), "Unknown")
    print(f"  {label:40s}  {score:.4f}")

print("Intermediate activation shape:", activations['layer4_block'].shape)
