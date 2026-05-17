import os
import warnings
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from unet import UNet
from diffusion import DDPM
from context_encoder import ContextEncoder
from skin_profile import extract_skin_profile
from mask_generator import get_region_masks

CHECKPOINT = "unet_makeup.pt"
PATCH_SIZE  = 64
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ────────────────────────────────────────────────────────────
def load_model():
    model   = UNet(ctx_dim=512).to(device)
    ddpm    = DDPM(T=1000, device=device)
    ctx_enc = ContextEncoder().to(device)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError("unet_makeup.pt not found. Run train.py first.")

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Model loaded from {CHECKPOINT}")
    return model, ddpm, ctx_enc

# ── Run inference on one image ────────────────────────────────────────────
def run_inference(image_path: str, prompt: str, region: str = "lips"):
    # Load image
    img_pil = Image.open(image_path).convert("RGB").resize((256, 256))
    img_np  = np.array(img_pil).astype(np.float32) / 255.0
    img_rgb = np.array(img_pil)

    # Step 1: Extract skin profile
    masks      = get_region_masks(img_rgb)
    skin_mask  = (masks["left_cheek"] | masks["right_cheek"]).astype(np.float32)
    profile    = extract_skin_profile(img_np, skin_mask)

    ita        = profile["ITA"]
    hue_angle  = profile["hue_angle"]
    print(f"Skin profile — ITA: {ita:.2f}°  Hue angle: {hue_angle:.2f}°")

    skin_vec = torch.tensor([[ita / 90.0, hue_angle / 180.0]],
                             dtype=torch.float32).to(device)

    # Step 2: Get region mask
    region_mask = masks.get(region, masks["lips"])
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        print(f"[!] No {region} region detected in image.")
        return None, img_pil, profile

    # Step 3: Crop region patch
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    crop    = img_pil.crop((x1, y1, x2, y2))
    patch_t = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])(crop).unsqueeze(0).to(device)

    # Step 4: Load model and run denoising
    model, ddpm, ctx_enc = load_model()
    with torch.no_grad():
        ctx       = ctx_enc([prompt], skin_vec)
        generated = ddpm.p_sample_loop(model, ctx, shape=(1, 3, PATCH_SIZE, PATCH_SIZE))

    # Step 5: Convert output tensor to image
    gen_np = (generated[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    gen_np = (gen_np * 255).astype(np.uint8)
    gen_pil = Image.fromarray(gen_np).resize((x2 - x1, y2 - y1))

    # Step 6: Blend patch back onto original face
    result = img_pil.copy()
    result_np = np.array(result)
    gen_region = np.array(gen_pil)

    region_only = np.zeros_like(result_np)
    region_only[y1:y2, x1:x2] = gen_region

    mask_3ch = np.stack([region_mask]*3, axis=-1).astype(np.float32)
    blended  = (result_np * (1 - mask_3ch) + region_only * mask_3ch).astype(np.uint8)
    blended_pil = Image.fromarray(blended)

    return gen_pil, blended_pil, profile


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_face.jpg"
    prompt     = sys.argv[2] if len(sys.argv) > 2 else "soft nude lipstick"
    region     = sys.argv[3] if len(sys.argv) > 3 else "lips"

    patch, blended, profile = run_inference(image_path, prompt, region)

    if blended is not None:
        blended.save("output_blended.jpg")
        patch.save("output_patch.jpg")
        print("Saved: output_blended.jpg")
        print("Saved: output_patch.jpg")