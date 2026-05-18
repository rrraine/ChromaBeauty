import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import UNet
from diffusion import DDPM
from context_encoder import ContextEncoder
from dataset import MakeupDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 150
LR = 1e-4
BATCH = 8

print(f"Training on {device}")

model = UNet(ctx_dim=512).to(device)

ddpm = DDPM(
    T=500,
    device=device
)

ctx_enc = ContextEncoder().to(device)

optimizer = torch.optim.AdamW(
    list(model.parameters()) +
    list(ctx_enc.skin_proj.parameters()),
    lr=LR
)

dataset = MakeupDataset(
    non_makeup_dir="data/non_makeup",
    makeup_dir="data/makeup",
)

loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=True,
    drop_last=True,
)

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    start = time.time()

    for batch in loader:

        x0 = batch["patch"].to(device)
        skin_vec = batch["skin_vec"].to(device)
        prompts = batch["prompt"]

        t = torch.randint(
            0,
            ddpm.T,
            (x0.shape[0],),
            device=device
        )

        xt, noise = ddpm.q_sample(x0, t)

        ctx = ctx_enc(prompts, skin_vec)

        pred_noise = model(xt, t, ctx)

        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    elapsed = time.time() - start

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "unet_makeup_best.pt"
        )

        print("Saved best model")

torch.save(
    model.state_dict(),
    "unet_makeup.pt"
)

print("Training complete")