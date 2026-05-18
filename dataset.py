from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch, numpy as np, os, random
from mask_generator import get_region_masks
from skin_profile import extract_skin_profile

MAKEUP_PROMPTS = [
    # Lip prompts
    "soft pink blush", "bold red lipstick", "everyday nude lipstick",
    "deep berry lips", "coral lip gloss", "warm peach blush",
    "rosy pink blush", "golden bronzer", "mauve lips and rose blush",
    "natural makeup look", "warm bronzer and peach blush",
    # Eyeshadow prompts
    "dark eyeshadow", "smoky eye", "neutral brown eyeshadow",
    "bold black eyeshadow", "soft pink eyeshadow", "golden shimmer eyeshadow",
    "purple eyeshadow", "bronze eye look", "earth tone eyeshadow",
    "glittery eye makeup", "dark dramatic eye",
]

def extract_lip_patch(img_pil, patch_size=64):
    """Extract and return lip region patch + mask from a PIL image."""
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    masks  = get_region_masks(np.array(img_pil))

    lip_mask = masks["lips"]
    ys, xs   = np.where(lip_mask > 0)

    if len(xs) == 0:
        patch = torch.zeros(3, patch_size, patch_size)
    else:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop  = img_pil.crop((x1, y1, x2, y2))
        patch = transforms.functional.resize(crop, (patch_size, patch_size))
        patch = transforms.functional.to_tensor(patch)
        patch = transforms.functional.normalize(patch, [0.5]*3, [0.5]*3)

    return patch, masks, img_np


class MakeupDataset(Dataset):
    """
    Loads paired non-makeup (input) and makeup (target) lip patches.
    - non_makeup_dir : source faces (skin profile extracted from here)
    - makeup_dir     : target makeup colors (model learns to generate these)
    - prompts        : optional {filename: prompt} mapping
    """
    def __init__(self, non_makeup_dir, makeup_dir, prompts: dict = {}, patch_size=64):
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
        self.prompts    = prompts
        self.patch_size = patch_size

        if not self.non_makeup_paths:
            raise FileNotFoundError(f"No images found in {non_makeup_dir}")
        if not self.makeup_paths:
            raise FileNotFoundError(f"No images found in {makeup_dir}")

        print(f"Dataset: {len(self.non_makeup_paths)} non-makeup, {len(self.makeup_paths)} makeup images")

    def __len__(self):
        return len(self.non_makeup_paths)

    def __getitem__(self, idx):
        #Non-makeup face (source — skin profile extracted from here)
        nm_path  = self.non_makeup_paths[idx]
        nm_fname = os.path.basename(nm_path)
        nm_img   = Image.open(nm_path).convert("RGB").resize((256, 256))
        nm_np    = np.array(nm_img).astype(np.float32) / 255.0

        nm_masks    = get_region_masks(np.array(nm_img))
        skin_mask   = (nm_masks["left_cheek"] | nm_masks["right_cheek"]).astype(np.float32)
        profile     = extract_skin_profile(nm_np, skin_mask)

        skin_vec = torch.tensor([
            profile["ITA"]       / 90.0,
            profile["hue_angle"] / 180.0,
        ], dtype=torch.float32)

        #Makeup face (target — model learns to generate this color)
        mk_path = random.choice(self.makeup_paths)
        mk_img  = Image.open(mk_path).convert("RGB").resize((256, 256))
        mk_patch, _, _ = extract_lip_patch(mk_img, self.patch_size)

        #Prompt
        prompt = self.prompts.get(nm_fname, random.choice(MAKEUP_PROMPTS))

        return {
            "patch":    mk_patch,   # target: makeup lip color
            "skin_vec": skin_vec,   # source: skin profile from non-makeup face
            "prompt":   prompt,
        }