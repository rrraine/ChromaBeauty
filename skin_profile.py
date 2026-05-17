import numpy as np
from skimage import color

def extract_skin_profile(face_rgb: np.ndarray, skin_mask: np.ndarray) -> dict:
    """
    face_rgb  : (H, W, 3) float32 in [0, 1]
    skin_mask : (H, W)    binary float32 — 1.0 on skin pixels
    Returns   : dict with ITA, hue_angle, mean_L, mean_a, mean_b
    """
    # 1. RGB → CIE L*a*b*
    lab = color.rgb2lab(face_rgb)          # shape (H, W, 3)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # 2. Mask to skin pixels only
    L_skin = L[skin_mask > 0.5]
    a_skin = a[skin_mask > 0.5]
    b_skin = b[skin_mask > 0.5]

    mL = float(np.mean(L_skin))
    ma = float(np.mean(a_skin))
    mb = float(np.mean(b_skin))

    # 3. ITA — light-to-dark axis
    ita = float(np.degrees(np.arctan((mL - 50) / (mb + 1e-6))))

    # 4. Hue angle — red vs yellow undertone
    hue_angle = float(np.degrees(np.arctan2(mb, ma)))

    return {"ITA": ita, "hue_angle": hue_angle, "L": mL, "a": ma, "b": mb}