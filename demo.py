import os
import warnings
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from inference import run_inference

# ── Demo config ───────────────────────────────────────────────────────────
# Replace these with actual face images from your dataset
SCENARIOS = [
    {
        "label":      "User A — cool/pink undertone",
        "image":      "data/non_makeup/xjpg_00000001.jpg",
        "prompt":     "soft pink blush",
        "region":     "left_cheek",
    },
    {
        "label":      "User B — warm/olive undertone",
        "image":      "data/non_makeup/xjpg_00000002.jpg",
        "prompt":     "soft pink blush",
        "region":     "left_cheek",
    },
]

def run_demo():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        'Skin-Hue Inversion Demo — Same prompt, different undertones',
        fontsize=14, fontweight='bold', y=1.01
    )

    gs = gridspec.GridSpec(2, len(SCENARIOS), figure=fig,
                           hspace=0.4, wspace=0.3)

    for col, sc in enumerate(SCENARIOS):
        print(f"\nRunning: {sc['label']}")
        print(f"Prompt : \"{sc['prompt']}\"")

        if not os.path.exists(sc["image"]):
            print(f"[!] Image not found: {sc['image']}")
            print("    Update SCENARIOS in demo.py with valid image paths.")
            continue

        patch, blended, profile = run_inference(
            sc["image"], sc["prompt"], sc["region"]
        )

        if blended is None:
            continue

        original = Image.open(sc["image"]).convert("RGB").resize((256, 256))

        # Row 0 — original face
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(original)
        ax0.set_title(f"{sc['label']}\nITA: {profile['ITA']:.1f}°  "
                      f"Hue: {profile['hue_angle']:.1f}°", fontsize=9)
        ax0.axis("off")

        # Row 1 — blended result
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(blended)
        ax1.set_title(f"Prompt: \"{sc['prompt']}\"", fontsize=9)
        ax1.axis("off")

    # Row labels
    fig.text(0.02, 0.75, "Original", va='center',
             rotation='vertical', fontsize=10, color='gray')
    fig.text(0.02, 0.25, "Generated", va='center',
             rotation='vertical', fontsize=10, color='gray')

    plt.savefig("demo_output.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: demo_output.png")

    # Also save loss curve if available
    _plot_loss_curve()

def _plot_loss_curve():
    loss_file = "losses.npy"
    if not os.path.exists(loss_file):
        print("No loss data found — skipping loss curve.")
        return