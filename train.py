import time
import torch
from torch.utils.data import DataLoader
from unet import UNet
from diffusion import DDPM
from context_encoder import ContextEncoder
from dataset import MakeupDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, LR, BATCH = 100, 1e-4, 4

print(f"Training on: {device}")
print(f"Epochs: {EPOCHS}  LR: {LR}  Batch: {BATCH}")

model   = UNet(ctx_dim=512).to(device)
ddpm    = DDPM(T=1000, device=device)
ctx_enc = ContextEncoder().to(device)
opt     = torch.optim.AdamW(
    list(model.parameters()) + list(ctx_enc.skin_proj.parameters()), lr=LR
)

dataset = MakeupDataset(
    non_makeup_dir="data/non_makeup",
    makeup_dir="data/makeup",
    prompts={}
)
loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=True)

best_loss   = float("inf")
losses      = []
epoch_times = []
start_total = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    epoch_loss  = 0

    for batch in loader:
        x0       = batch["patch"].to(device)      # (B, 3, 64, 64)
        skin_vec = batch["skin_vec"].to(device)   # (B, 2)
        prompts  = batch["prompt"]

        t         = torch.randint(0, ddpm.T, (x0.shape[0],), device=device)
        xt, noise = ddpm.q_sample(x0, t)

        ctx      = ctx_enc(prompts, skin_vec)       # (B, 78, 512)
        eps_pred = model(xt, t, ctx)                # ← t now passed to model

        loss = torch.nn.functional.mse_loss(eps_pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    avg        = epoch_loss / len(loader)
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    losses.append(avg)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    epochs_left    = EPOCHS - (epoch + 1)
    eta_seconds    = avg_epoch_time * epochs_left
    eta_mins       = int(eta_seconds // 60)
    eta_secs       = int(eta_seconds % 60)
    elapsed        = time.time() - start_total
    elapsed_mins   = int(elapsed // 60)
    elapsed_secs   = int(elapsed % 60)

    print(
        f"Epoch {epoch+1:>3}/{EPOCHS}  "
        f"Loss: {avg:.4f}  "
        f"Epoch time: {epoch_time:.1f}s  "
        f"Elapsed: {elapsed_mins}m {elapsed_secs}s  "
        f"ETA: {eta_mins}m {eta_secs}s"
    )

    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "unet_makeup_best.pt")

torch.save(model.state_dict(), "unet_makeup.pt")
total = time.time() - start_total
print(f"\nDone! Total time: {int(total//60)}m {int(total%60)}s  Best loss: {best_loss:.4f}")
print("Saved: unet_makeup.pt and unet_makeup_best.pt")