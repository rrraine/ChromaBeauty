import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from PIL import Image

from diffusion import DDPM
from unet import UNet
from context_encoder import ContextEncoder
from mask_generator import get_region_masks
from skin_profile import extract_skin_profile

#(dynamic or default)
prompt = sys.argv[1] if len(sys.argv) > 1 else "soft pink blush"
print(f'Prompt: "{prompt}"')

#Setup
device  = "cuda" if torch.cuda.is_available() else "cpu"
model   = UNet(ctx_dim=512).to(device)
ddpm    = DDPM(T=1000, device=device)
ctx_enc = ContextEncoder().to(device)

model.load_state_dict(torch.load("unet_makeup_best.pt", map_location=device))
model.eval()

#Pick a random face from dataset
NON_MAKEUP_DIR = "data/non_makeup"
images = sorted([f for f in os.listdir(NON_MAKEUP_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
if not images:
    raise FileNotFoundError("No images found in data/non_makeup/")

face_path = os.path.join(NON_MAKEUP_DIR, random.choice(images))
print(f"Using face: {os.path.basename(face_path)}")
face_pil  = Image.open(face_path).convert("RGB").resize((256, 256))
face_np   = np.array(face_pil).astype(np.float32) / 255.0

#Extract skin profile from the real face
masks     = get_region_masks(np.array(face_pil))
skin_mask = (masks["left_cheek"] | masks["right_cheek"]).astype(np.float32)
profile   = extract_skin_profile(face_np, skin_mask)

skin_vec  = torch.tensor([[
    profile["ITA"]       / 90.0,
    profile["hue_angle"] / 180.0,
]], dtype=torch.float32).to(device)

ita = profile["ITA"]
print(f"Skin profile — ITA: {ita:.1f}°  Hue: {profile['hue_angle']:.1f}°")

#Determine skin tone label from ITA
if ita > 55:
    skin_tone_label = "Very Light"
elif ita > 41:
    skin_tone_label = "Light"
elif ita > 28:
    skin_tone_label = "Intermediate"
elif ita > 10:
    skin_tone_label = "Tan"
elif ita > -30:
    skin_tone_label = "Brown"
else:
    skin_tone_label = "Dark"

print(f"Skin tone: {skin_tone_label}")

#Generate color patch from prompt
ctx      = ctx_enc([prompt], skin_vec)
patch    = ddpm.p_sample_loop(model, ctx, shape=(1, 3, 64, 64))
patch_np = (patch[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

#Adjust color tone based on ITA skin tone threshold
if ita > 41:        # light skin — lift color toward lighter/pastel
    tone_factor = 1.3
elif ita > 10:      # medium skin — slight lift
    tone_factor = 1.1
elif ita > -30:     # tan/brown skin — keep natural
    tone_factor = 0.95
else:               # dark skin — deepen color slightly
    tone_factor = 0.85

mean_color = (patch_np.reshape(-1, 3).mean(axis=0) * tone_factor).clip(0, 1)

#Force dark color for dark/smoky prompts
prompt_lower = prompt.lower()
if "dark" in prompt_lower or "smoky" in prompt_lower or "bold black" in prompt_lower:
    mean_color = (mean_color * 0.3).clip(0, 1)

#Determine regions based on prompt keywords
regions = []
if any(w in prompt_lower for w in ["lip", "lipstick", "gloss", "mouth"]):
    regions.append(("lips", 0.45))
if any(w in prompt_lower for w in ["blush", "cheek", "bronzer", "contour", "highlight"]):
    regions.append(("left_cheek", 0.25))
    regions.append(("right_cheek", 0.25))
if any(w in prompt_lower for w in ["eyeshadow", "eye", "eyelid"]):
    regions.append(("left_eye", 0.6))
    regions.append(("right_eye", 0.6))
if not regions:
    # Default: apply to all regions if prompt is ambiguous
    regions = [("lips", 0.45), ("left_cheek", 0.25), ("right_cheek", 0.25)]

#Overlay patch color onto detected regions
overlay = face_np.copy()
for region, strength in regions:
    mask = masks[region].astype(bool)
    if mask.any():
        overlay[mask] = (face_np[mask] * (1 - strength) + mean_color * strength).clip(0, 1)

#Plot
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#1a1a1a")

fig.text(0.5, 0.97, "ChromaBeauty — Skin-Adaptive Makeup",
         ha="center", va="top", fontsize=16, fontweight="bold", color="white")
fig.text(0.5, 0.91, f'Prompt: "{prompt}"',
         ha="center", va="top", fontsize=12, color="#e0a0c0", style="italic")

gs = gridspec.GridSpec(1, 3, left=0.04, right=0.96, top=0.85, bottom=0.12, wspace=0.06)

panels = [
    (face_np,  "Original Face",          f"(no makeup) — {skin_tone_label} skin"),
    (overlay,  "With Makeup Applied",    f'"{prompt}"'),
    (patch_np, "Generated Color Swatch", f"ITA={ita:.1f}°  Hue={profile['hue_angle']:.1f}°"),
]

for i, (img, title, subtitle) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
    fig.text(
        0.04 + i * (0.92 / 3) + (0.92 / 6),
        0.09,
        subtitle,
        ha="center", va="top",
        fontsize=9, color="#aaaaaa"
    )

#Auto-increment filename
i = 1
while os.path.exists(f"comparison_output_{i}.png"):
    i += 1
filename = f"comparison_output_{i}.png"

plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
print(f"Saved {filename}")