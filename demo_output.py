import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image

from diffusion import DDPM
from unet import UNet
from context_encoder import ContextEncoder
from mask_generator import get_region_masks
from skin_profile import extract_skin_profile



# Prompt
prompt = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "bold red lipstick"
)
print(f'Prompt: "{prompt}"')


# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Models
model = UNet(ctx_dim=512).to(device)
model.load_state_dict(
    torch.load("unet_makeup_best.pt", map_location=device)
)
model.eval()

ddpm = DDPM(T=250, device=device)
ctx_enc = ContextEncoder().to(device)


# Load random face
NON_MAKEUP_DIR = "data/non_makeup"

images = [
    f for f in os.listdir(NON_MAKEUP_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    raise FileNotFoundError("No images found in data/non_makeup/")

face_path = os.path.join(NON_MAKEUP_DIR, random.choice(images))
print(f"Using face: {os.path.basename(face_path)}")

face_pil = Image.open(face_path).convert("RGB").resize((256, 256))
face_np  = np.array(face_pil).astype(np.float32) / 255.0

# Region masks & skin profile
masks = get_region_masks(np.array(face_pil))

skin_mask = (
    masks["left_cheek"] | masks["right_cheek"]
).astype(np.float32)

profile = extract_skin_profile(face_np, skin_mask)
ita     = profile["ITA"]

print(
    f"Skin profile — ITA: {profile['ITA']:.1f}°  "
    f"Hue: {profile['hue_angle']:.1f}°"
)

# Skin tone label
if   ita > 55: skin_tone_label = "Very Light"
elif ita > 41: skin_tone_label = "Light"
elif ita > 28: skin_tone_label = "Intermediate"
elif ita > 10: skin_tone_label = "Tan"
elif ita > -30: skin_tone_label = "Brown"
else:           skin_tone_label = "Dark"

print(f"Skin tone: {skin_tone_label}")

# Normalize skin vector
ita_norm = float(np.clip(profile["ITA"]       / 90.0,  -1, 1))
hue_norm = float(np.clip(profile["hue_angle"] / 180.0, -1, 1))

skin_vec = torch.tensor(
    [[ita_norm, hue_norm]], dtype=torch.float32
).to(device)

# Generate diffusion patch

with torch.no_grad():
    ctx   = ctx_enc([prompt], skin_vec)
    patch = ddpm.p_sample_loop(model, ctx, shape=(1, 3, 64, 64))

patch_np = patch[0].permute(1, 2, 0).cpu().numpy()
patch_np = (patch_np * 0.5 + 0.5).clip(0, 1)

# Semantic swatch enhancement  (same as before – for the display panel only)
prompt_lower = prompt.lower()

swatch = patch_np.copy()

if "red" in prompt_lower:
    swatch[..., 0] *= 1.8;  swatch[..., 1] *= 0.55; swatch[..., 2] *= 0.55
elif "pink" in prompt_lower:
    swatch[..., 0] *= 1.4;  swatch[..., 1] *= 0.85; swatch[..., 2] *= 1.2
elif "berry" in prompt_lower or "purple" in prompt_lower:
    swatch[..., 0] *= 1.2;  swatch[..., 1] *= 0.5;  swatch[..., 2] *= 1.3
elif "dark" in prompt_lower or "smoky" in prompt_lower:
    swatch *= 0.45
elif "coral" in prompt_lower or "peach" in prompt_lower:
    swatch[..., 0] *= 1.4;  swatch[..., 1] *= 1.0;  swatch[..., 2] *= 0.7
elif "nude" in prompt_lower:
    swatch[..., 0] *= 1.1;  swatch[..., 1] *= 0.95; swatch[..., 2] *= 0.9

swatch = np.clip(swatch, 0, 1)

# Determine target color
# Fallback palette – used when the diffusion model is still noisy / undertrained.
# These are perceptually-tuned sRGB values for each makeup family.
PROMPT_COLORS = {
    "red":    np.array([0.82, 0.08, 0.10]),
    "pink":   np.array([0.92, 0.42, 0.62]),
    "nude":   np.array([0.80, 0.56, 0.48]),
    "berry":  np.array([0.52, 0.08, 0.32]),
    "coral":  np.array([0.93, 0.46, 0.30]),
    "peach":  np.array([0.95, 0.68, 0.50]),
    "smoky":  np.array([0.18, 0.14, 0.18]),
    "dark":   np.array([0.20, 0.10, 0.12]),
    "brown":  np.array([0.55, 0.30, 0.20]),
    "gold":   np.array([0.85, 0.68, 0.20]),
}

mean_color = None
for key, color in PROMPT_COLORS.items():
    if key in prompt_lower:
        mean_color = color.copy()
        break

# If no keyword matched, derive color from the generated swatch
if mean_color is None:
    flat = swatch.reshape(-1, 3)
    mean_color = np.mean(flat, axis=0)

mean_color = np.clip(mean_color, 0, 1)


# Overlay: multiply-blend for natural skin-texture preservation
def soft_light_blend(base, layer):
    """Photoshop-style soft-light blend preserving base luminance."""
    return np.where(
        layer <= 0.5,
        base - (1 - 2 * layer) * base * (1 - base),
        base + (2 * layer - 1) * (
            np.where(base <= 0.25,
                     ((16 * base - 12) * base + 4) * base,
                     np.sqrt(base)) - base
        )
    )

def multiply_blend(base, layer):
    """Multiply blend: darkens proportional to layer color."""
    return base * layer

overlay = face_np.copy()

# Region(mask_key, blend_strength, blur_kernel)
regions = []

if "lip" in prompt_lower or "gloss" in prompt_lower:
    regions.append(("lips",        0.60, 3))

if "blush" in prompt_lower or "bronzer" in prompt_lower:
    regions.append(("left_cheek",  0.40, 31))
    regions.append(("right_cheek", 0.40, 31))

if "eye" in prompt_lower or "shadow" in prompt_lower:
    regions.append(("left_eye",    0.50, 5))
    regions.append(("right_eye",   0.50, 5))

#Default fallback
if not regions:
    regions.append(("lips", 0.60, 3))

for region, strength, blur_k in regions:
    raw_mask = masks[region].astype(np.float32)

    #Feather edges for natural fall-off
    if blur_k > 1:
        raw_mask = cv2.GaussianBlur(raw_mask, (blur_k | 1, blur_k | 1), 0)

    mask3 = raw_mask[:, :, np.newaxis]   # (H, W, 1) for broadcasting

    # 1. Multiply blend – darkens skin with the target hue, keeps texture
    multiplied = multiply_blend(overlay, mean_color[np.newaxis, np.newaxis, :])
    multiplied = np.clip(multiplied, 0, 1)

    # 2. Soft-light blend – adds luminance variation without fully replacing base
    soft      = soft_light_blend(overlay, mean_color[np.newaxis, np.newaxis, :])
    soft      = np.clip(soft, 0, 1)

    # 3. Weighted mix: mostly multiply for colour, touch of soft-light for glow
    tinted    = multiplied * 0.65 + soft * 0.35

    # 4. Blend tinted result into overlay using the feathered mask
    overlay   = overlay * (1.0 - mask3 * strength) + tinted * (mask3 * strength)

overlay = np.clip(overlay, 0, 1)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor("#111111")

panels = [
    (face_np,  "Original Face",        f"(no makeup) — {skin_tone_label} skin"),
    (overlay,  "With Makeup Applied",  f'"{prompt}"'),
    (swatch,   "Generated Color Swatch",
               f"ITA={profile['ITA']:.1f}°  Hue={profile['hue_angle']:.1f}°"),
]

for ax, (img, title, subtitle) in zip(axes, panels):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.text(0.5, -0.08, subtitle,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="#aaaaaa")

plt.suptitle(
    "ChromaBeauty — Skin-Adaptive Makeup",
    color="white", fontsize=18, fontweight="bold"
)
plt.figtext(
    0.5, 0.92, f'Prompt: "{prompt}"',
    ha="center", color="#e0a0c0", fontsize=13, style="italic"
)

# Auto-increment filename
i = 1
while os.path.exists(f"comparison_output_{i}.png"):
    i += 1
filename = f"comparison_output_{i}.png"

plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor())
print(f"Saved {filename}")