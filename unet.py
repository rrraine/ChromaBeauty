import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        emb = t[:, None].float() * freq[None]           # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1) # (B, dim)


class CrossAttention(nn.Module):
    """Spatial cross-attention: image features attend to [text + skin vector]."""
    def __init__(self, feat_dim, ctx_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (feat_dim // heads) ** -0.5
        self.q   = nn.Linear(feat_dim, feat_dim)
        self.k   = nn.Linear(ctx_dim,  feat_dim)
        self.v   = nn.Linear(ctx_dim,  feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, ctx):
        # x   : (B, C, H, W)
        # ctx : (B, seq, ctx_dim)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        Q = self.q(x_flat)
        K = self.k(ctx)
        V = self.v(ctx)

        def split(t):
            return t.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)

        Q, K, V = split(Q), split(K), split(V)
        attn = torch.softmax(Q @ K.transpose(-1, -2) * self.scale, dim=-1)
        out  = (attn @ V).permute(0, 2, 1, 3).reshape(B, H * W, C)
        out  = self.out(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out + x  # residual


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        # Project time embedding to channel width if provided
        self.time_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None else None
        )

    def forward(self, x, temb=None):
        h = self.net(x)
        if self.time_proj is not None and temb is not None:
            h = h + self.time_proj(temb)[:, :, None, None]
        return h


class UNet(nn.Module):
    """4-level U-Net with timestep conditioning via sinusoidal embeddings."""
    def __init__(self, ctx_dim=512):
        super().__init__()
        chs = [64, 128, 256, 512]
        time_dim = chs[0] * 4  # 256

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmb(chs[0]),
            nn.Linear(chs[0], time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder — every ConvBlock receives the time embedding
        self.enc  = nn.ModuleList([
            ConvBlock(3 if i == 0 else chs[i - 1], chs[i], time_emb_dim=time_dim)
            for i in range(4)
        ])
        self.pool = nn.MaxPool2d(2)

        # Cross-attention at bottleneck (8×8)
        self.attn = CrossAttention(chs[-1], ctx_dim)

        # Decoder
        self.up  = nn.ModuleList([
            nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2)
            for i in range(3, 0, -1)
        ])
        self.dec = nn.ModuleList([
            ConvBlock(chs[i], chs[i - 1], time_emb_dim=time_dim)
            for i in range(3, 0, -1)
        ])

        self.head = nn.Conv2d(chs[0], 3, 1)  # predict noise ε

    def forward(self, x, t, ctx):
        """
        x   : (B, 3, 64, 64)  noisy patch
        t   : (B,)             integer timesteps
        ctx : (B, seq, ctx_dim)
        """
        temb = self.time_mlp(t)   # (B, time_dim)

        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x, temb)
            if i < 3:
                skips.append(x)
                x = self.pool(x)

        x = self.attn(x, ctx)     # cross-attention at 8×8 bottleneck

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1), temb)

        return self.head(x)