import torch
from torch.utils.data import DataLoader
from unet import UNet
from diffusion import DDPM
from context_encoder import ContextEncoder
from dataset import MakeupDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, LR, BATCH = 50, 1e-4, 8

model   = UNet(ctx_dim=512).to(device)
ddpm    = DDPM(T=1000, device=device)
ctx_enc = ContextEncoder().to(device)
opt     = torch.optim.AdamW(
    list(model.parameters()) + list(ctx_enc.skin_proj.parameters()), lr=LR)

# Load your dataset here (MT-Dataset recommended)
dataset = MakeupDataset("data/non_makeup", prompts={})
loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch in loader:
        x0       = batch["patch"].to(device)          # (B,3,64,64)
        skin_vec = batch["skin_vec"].to(device)       # (B,2)
        prompts  = batch["prompt"]

        t     = torch.randint(0, ddpm.T, (x0.shape[0],), device=device)
        xt, noise = ddpm.q_sample(x0, t)

        ctx      = ctx_enc(prompts, skin_vec)          # (B,78,512)
        eps_pred = model(xt, ctx)

        loss = torch.nn.functional.mse_loss(eps_pred, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(loader)
    losses.append(avg)
    print(f"Epoch {epoch+1}/{EPOCHS}  MSE loss: {avg:.4f}")

torch.save(model.state_dict(), "unet_makeup.pt")