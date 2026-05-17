from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch, numpy as np, os
from mask_generator import get_region_masks
from skin_profile import extract_skin_profile

class MakeupDataset(Dataset):
    def __init__(self, image_dir, prompts: dict, patch_size=64):
        self.paths   = sorted([os.path.join(image_dir, f)
                                for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.prompts = prompts      # {filename: "everyday nude lipstick"}
        self.resize  = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        self.patch_size = patch_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        fname = os.path.basename(path)
        img   = Image.open(path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0

        # --- Math labels (computed on-the-fly) ---
        masks = get_region_masks(np.array(img))
        skin_mask = (masks["left_cheek"] | masks["right_cheek"]).astype(np.float32)
        profile   = extract_skin_profile(img_np, skin_mask)

        # Normalize skin vector to [-1, 1]  (ITA ∈ [-90,90], hue ∈ [-180,180])
        skin_vec = torch.tensor([
            profile["ITA"]       / 90.0,
            profile["hue_angle"] / 180.0,
        ], dtype=torch.float32)

        # --- Lip patch (64×64) ---
        lip_mask  = masks["lips"]
        ys, xs    = np.where(lip_mask > 0)
        if len(xs) == 0:
            patch = torch.zeros(3, self.patch_size, self.patch_size)
        else:
            x1,y1,x2,y2 = xs.min(),ys.min(),xs.max(),ys.max()
            crop = img.crop((x1, y1, x2, y2))
            patch = transforms.functional.resize(crop, (self.patch_size, self.patch_size))
            patch = transforms.functional.to_tensor(patch)
            patch = transforms.functional.normalize(patch, [0.5]*3, [0.5]*3)

        prompt = self.prompts.get(fname, "natural makeup look")
        return {"patch": patch, "skin_vec": skin_vec, "prompt": prompt}