import torch
import torchvision
from torchvision.io import read_video
import json 
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import numpy as np
from torchvision.models.video import R3D_18_Weights

# Load json with class mapping (class name → index)
with open("kinetics_classnames.json", "r") as f:
    old_mapping = json.load(f)

# Invert it: index (as string) → class name (no extra quotes)
new_mapping = {str(v): k.replace('"', '') for k, v in old_mapping.items()}

# Save it as kinetics_classnames1.json
with open("kinetics_classnames1.json", "w") as f:
    json.dump(new_mapping, f, indent=2)

print("Done! You now have a correct index → class name JSON.")


# Loads new json
with open("kinetics_classnames1.json", "r") as f:
    id_to_class = json.load(f)

# loads model, eval for inference
model = torchvision.models.video.r3d_18(weights="DEFAULT")
model = model.eval()

# loads video clip
video_path = "/Users/fleurconway/Documents/Programming/Kinetics_400_First_Test/Baking_Christmas_Cookies_Clip.mp4"
frames, _, _ = read_video(video_path, pts_unit='sec')

# --- Preprocess ---
frames = frames.permute(0, 3, 1, 2)            # [T, C, H, W]
# frames = frames[:16].float() / 255.0          # first 16 frames, normalise
T = frames.shape[0]
idx = torch.linspace(0, T-1, 16).long()
frames = frames[idx]

#convert to float
frames = frames.float() /255.0 #[T. C, H, W]

#resize, short edge = 128 and keep aspect ratio
_, C, H, W = frames.shape
short_edge = 128

if H < W:
    new_h = short_edge
    new_w = int(W * short_edge /H)
else:
    new_w = short_edge
    new_h = int( H * short_edge/W)

frames = F.interpolate(
    frames,
    size=(new_h, new_w),
    mode="bilinear",
    align_corners=False
)

#now centre the crop to 112x112

top = (new_h - 112) // 2
left = (new_w - 112) // 2
frames = frames[:, :, top:top+112, left:left+112]

#normalise
mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
frames = (frames - mean) / std

#add batch and reorder for 3D CNN
frames = frames.unsqueeze(0)                  # [1, T, C, H, W] as neural networks always expect batches even of size 1
frames = frames.permute(0, 2, 1, 3, 4)         # [B, C, T, H, W] reorder for 3D CNN input

# --- Forward pass + predictions ---
preds = model(frames)

probs = torch.softmax(preds, dim=1)[0]
top5 = torch.topk(probs, 5)
 
print("Top‑5 predictions:")
for idx, score in zip(top5.indices, top5.values):
    label = id_to_class.get(str(int(idx)), "Unknown")
    print(f"  {label:40s}  {score:.4f}")

print("Intermediate activation shape:", activations['layer4_block'].shape)
print(R3D_18_Weights.DEFAULT.transforms())

class TCAVWrapper:
    def __init__(self, model, bottleneck_layer_name, id_to_class):
        self.model = model.eval()
        self.bottleneck_layer_name= bottleneck_layer_name
        self.id_to_class = id_to_class
        self.activations = {}

        #register the forward hook
        layer = dict([*self.model.named_modules()])[bottleneck_layer_name]
        layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        #this pulls the activations from the layer I think
        self.activations[self.bottleneck_layer_name] = output
        output.retain_grad()

    def get_activations_and_grads(self, x, target_class):
        #at zero gradient, find the predictions of? need to ask chatgpt these ones really
        self.model.zero_grad()
        preds = self.model(x)
        preds[0, target_class].backward()
        acts = self.activations[self.bottleneck_layer_name]
        grads = acts.grad
        return acts.detach(), grads.detach(), preds.detach()

    def label_to_index(self, label):
        for idx, name in self.id_to_class.items():
            if name == label:
                return int(idx)
        raise ValueError(f"Label {label} not found.")
    
wrapper = TCAVWrapper(model, bottleneck_layer_name='layer4.1.conv2', id_to_class=id_to_class)

concept_videos = [frames] #just to check this all works rn

frames_flipped = torch.flip(frames, dims=[4])  # flip width
random_videos = [frames_flipped]

target_label = 'baking cookies'
target_class_idx = wrapper.label_to_index(target_label)

def flatten_acts(acts):
    #flatten CxTxHxW -> 1D vector
    return acts.reshape(acts.shape[0], -1)

#collect activations:

concept_acts = torch.cat([flatten_acts(wrapper.get_activations_and_grads(v, target_class_idx)[0])
                          for v in concept_videos], dim=0)
random_acts = torch.cat([flatten_acts(wrapper.get_activations_and_grads(v, target_class_idx)[0])
                         for v in random_videos], dim=0)

#train a simple CAV
X = torch.cat([concept_acts, random_acts], dim=0).numpy()
y = np.array([1]*concept_acts.shape[0] + [0]*random_acts.shape[0])

cav = LogisticRegression().fit(X, y).coef_.flatten() #concept vector!

#compute direction derivative i.e. TCAV score - in this case the same clip as before
acts, grads, _ = wrapper.get_activations_and_grads(frames, target_class_idx)
acts_flat = flatten_acts(acts).numpy()
grads_flat = grads.reshape(grads.shape[0], -1).numpy()

#directional derivative along CAV
dd = np.dot(grads_flat, cav)
tcav_score = np.mean(dd >0) # fraction of positive alignment
print(f"TCAV score for '{target_label}' along this concept:", tcav_score)