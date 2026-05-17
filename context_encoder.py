import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class ContextEncoder(nn.Module):
    """
    Combines CLIP text embeddings with the 2D skin vector (ITA, hue_angle).
    Output: (B, seq+1, 512) context for cross-attention.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_enc  = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # Freeze CLIP — we only train U-Net
        for p in self.text_enc.parameters():
            p.requires_grad = False

        self.skin_proj = nn.Linear(2, 512)   # [ITA, hue_angle] → 512-d token

    def forward(self, prompts: list[str], skin_vecs: torch.Tensor):
        # skin_vecs: (B, 2)  — [ITA, hue_angle] normalized to [-1, 1]
        tokens = self.tokenizer(prompts, return_tensors="pt",
                                padding=True, truncation=True, max_length=77)
        tokens = {k: v.to(skin_vecs.device) for k,v in tokens.items()}
        text_emb = self.text_enc(**tokens).last_hidden_state   # (B, 77, 512)

        skin_tok = self.skin_proj(skin_vecs).unsqueeze(1)      # (B, 1, 512)
        return torch.cat([text_emb, skin_tok], dim=1)          # (B, 78, 512)