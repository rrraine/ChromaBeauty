import matplotlib.pyplot as plt, torch
from diffusion import DDPM
from unet import UNet
from context_encoder import ContextEncoder

model.load_state_dict(torch.load("unet_makeup.pt"))
model.eval()

scenarios = [
    {"label": "User A (cool/pink undertone)", "ita": 45.0, "hue": 35.0},
    {"label": "User B (warm/olive undertone)", "ita": 45.0, "hue": 52.0},
]
prompt = "soft pink blush"

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, sc in zip(axes, scenarios):
    skin_vec = torch.tensor([[sc["ita"]/90.0, sc["hue"]/180.0]])
    ctx = ctx_enc([prompt], skin_vec.to(device))
    out = ddpm.p_sample_loop(model, ctx, shape=(1,3,64,64))
    patch = (out[0].permute(1,2,0).cpu().numpy() * 0.5 + 0.5).clip(0,1)
    ax.imshow(patch)
    ax.set_title(f"{sc['label']}\nITA={sc['ita']}° Hue={sc['hue']}°", fontsize=9)
    ax.axis("off")
plt.suptitle(f'Prompt: "{prompt}"', fontweight="bold")
plt.tight_layout(); plt.savefig("comparison_output.png", dpi=150)