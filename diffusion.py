import torch
import torch.nn.functional as F

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas      = betas
        self.alpha_bar  = alpha_bar
        self.sqrt_ab    = alpha_bar.sqrt()
        self.sqrt_1mab  = (1 - alpha_bar).sqrt()

    def q_sample(self, x0, t, noise=None):
        """Forward: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.sqrt_ab[t].view(-1,1,1,1)
        return ab * x0 + self.sqrt_1mab[t].view(-1,1,1,1) * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, ctx, shape):
        """Reverse: denoise from pure noise → clean patch."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            eps_pred = model(x, ctx)
            # DDPM posterior mean
            ab  = self.alpha_bar[t]
            ab1 = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
            beta = self.betas[t]
            x0_pred = (x - self.sqrt_1mab[t] * eps_pred) / (ab.sqrt() + 1e-8)
            x0_pred = x0_pred.clamp(-1, 1)
            mean = (ab1.sqrt() * beta / (1-ab)) * x0_pred \
                 + ((1-beta).sqrt() * (1-ab1) / (1-ab)) * x
            noise = torch.randn_like(x) if t > 0 else 0
            x = mean + ((beta * (1-ab1)/(1-ab)).sqrt()) * noise
        return x