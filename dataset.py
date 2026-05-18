from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os

from mask_generator import get_region_masks
from skin_profile import extract_skin_profile


PROMPT_KEYWORDS = {
    "red": "bold red lipstick",
    "pink": "soft pink blush",
    "nude": "everyday nude lipstick",
    "berry": "deep berry lips",
    "coral": "coral lip gloss",
    "smoky": "smoky eye",
    "brown": "neutral brown eyeshadow",
    "gold": "golden shimmer eyeshadow",
}


class MakeupDataset(Dataset):
    def __init__(self, non_makeup_dir, makeup_dir, patch_size=64):
        self.non_makeup_paths = sorted([
            os.path.join(non_makeup_dir, f)
            for f in os.listdir(non_makeup_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.makeup_paths = sorted([
            os.path.join(makeup_dir, f)
            for f in os.listdir(makeup_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        self.patch_size = patch_size

        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        print(f"Dataset loaded: {len(self.non_makeup_paths)} source images")
        print(f"Dataset loaded: {len(self.makeup_paths)} makeup images")

    def __len__(self):
        return min(len(self.non_makeup_paths), len(self.makeup_paths))

    def infer_prompt(self, filename):
        name = filename.lower()

        for key, prompt in PROMPT_KEYWORDS.items():
            if key in name:
                return prompt

        return "natural makeup look"

    def extract_region_patch(self, img_pil, region_name):
        img_np = np.array(img_pil)
        masks = get_region_masks(img_np)

        if region_name not in masks:
            return torch.zeros(3, self.patch_size, self.patch_size)

        mask = masks[region_name]
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return torch.zeros(3, self.patch_size, self.patch_size)

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        crop = img_pil.crop((x1, y1, x2, y2))

        return self.transform(crop)

    def __getitem__(self, idx):
        nm_path = self.non_makeup_paths[idx]
        mk_path = self.makeup_paths[idx]

        nm_img = Image.open(nm_path).convert("RGB").resize((256, 256))
        mk_img = Image.open(mk_path).convert("RGB").resize((256, 256))

        nm_np = np.array(nm_img).astype(np.float32) / 255.0

        masks = get_region_masks(np.array(nm_img))

        skin_mask = (
            masks["left_cheek"] |
            masks["right_cheek"]
        ).astype(np.float32)

        profile = extract_skin_profile(nm_np, skin_mask)

        ita_norm = np.clip(profile["ITA"] / 90.0, -1, 1)
        hue_norm = np.clip(profile["hue_angle"] / 180.0, -1, 1)

        skin_vec = torch.tensor([
            ita_norm,
            hue_norm,
        ], dtype=torch.float32)

        filename = os.path.basename(mk_path)
        prompt = self.infer_prompt(filename)

        prompt_lower = prompt.lower()

        if "lip" in prompt_lower or "gloss" in prompt_lower:
            region = "lips"

        elif "blush" in prompt_lower or "bronzer" in prompt_lower:
            region = "left_cheek"

        elif "eye" in prompt_lower or "shadow" in prompt_lower:
            region = "left_eye"

        else:
            region = "lips"

        patch = self.extract_region_patch(mk_img, region)

        return {
            "patch": patch,
            "skin_vec": skin_vec,
            "prompt": prompt,
            "region": region,
        }