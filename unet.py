import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """Spatial cross-attention: image features attend to [text + skin vector]."""
    def __init__(self, feat_dim, ctx_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale  = (feat_dim // heads) ** -0.5
        self.q = nn.Linear(feat_dim, feat_dim)
        self.k = nn.Linear(ctx_dim,  feat_dim)
        self.v = nn.Linear(ctx_dim,  feat_dim)
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, x, ctx):
        # x   : (B, C, H, W)  — spatial feature map
        # ctx : (B, seq, ctx_dim) — [text tokens | skin vector token]
        B, C, H, W = x.shape
        x_flat = x.permute(0,2,3,1).reshape(B, H*W, C)   # (B, HW, C)
        Q = self.q(x_flat)
        K = self.k(ctx)
        V = self.v(ctx)
        # Split heads
        def split(t): return t.reshape(B, -1, self.heads, C//self.heads).permute(0,2,1,3)
        Q, K, V = split(Q), split(K), split(V)
        attn = torch.softmax(Q @ K.transpose(-1,-2) * self.scale, dim=-1)
        out  = (attn @ V).permute(0,2,1,3).reshape(B, H*W, C)
        out  = self.out(out).reshape(B, H, W, C).permute(0,3,1,2)
        return out + x   # residual

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    """4-level U-Net. Input: noisy patch (B,3,64,64) + context."""
    def __init__(self, ctx_dim=512):
        super().__init__()
        chs = [64, 128, 256, 512]

        # Encoder
        self.enc = nn.ModuleList([ConvBlock(3 if i==0 else chs[i-1], chs[i]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)

        # Cross-attention at bottleneck
        self.attn = CrossAttention(chs[-1], ctx_dim)

        # Decoder
        self.up   = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i-1], 2, stride=2) for i in range(3,0,-1)])
        self.dec  = nn.ModuleList([ConvBlock(chs[i], chs[i-1]) for i in range(3,0,-1)])

        self.head = nn.Conv2d(chs[0], 3, 1)   # predict noise ε

    def forward(self, x, ctx):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 3:
                skips.append(x)
                x = self.pool(x)

        x = self.attn(x, ctx)     # cross-attention at 8×8 bottleneck

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.head(x)