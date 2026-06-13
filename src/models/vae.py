"""
vae.py
Conditional Variational Autoencoder for rare-disease data augmentation.
Trained offline (Phase 2) on rare-class symptom embeddings.
Generates synthetic symptom vectors conditioned on disease label.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_LATENT_DIM  = get("vae.latent_dim", 64)
_HIDDEN_DIMS = get("vae.hidden_dims", [256, 128])
_OUTPUT_DIR  = Path(get("vae.output_dir", "models/vae"))


class ConditionalVAE(nn.Module):
    """
    Conditional VAE:
      - Encoder maps (x, label_onehot) -> (mu, log_var)
      - Reparameterisation trick samples z
      - Decoder maps (z, label_onehot) -> x_reconstructed
    Input x: symptom embedding vector (e.g. GloVe 100-dim)
    Conditioning: one-hot disease label
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim   = input_dim
        self.num_classes = num_classes
        enc_input = input_dim + num_classes

        # Encoder
        enc_layers = []
        prev = enc_input
        for h in _HIDDEN_DIMS:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu     = nn.Linear(prev, _LATENT_DIM)
        self.fc_logvar = nn.Linear(prev, _LATENT_DIM)

        # Decoder
        dec_layers = []
        prev = _LATENT_DIM + num_classes
        for h in reversed(_HIDDEN_DIMS):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h       = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, log_var = self.encode(x, c)
        z           = self.reparameterise(mu, log_var)
        x_recon     = self.decode(z, c)
        return x_recon, mu, log_var

    # ── Loss ──────────────────────────────────────────────────────────────────
    @staticmethod
    def loss(x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor,
             kl_weight: float = 0.5) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_weight * kl_loss

    # ── Generation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, disease_id: int, n_samples: int = 1) -> torch.Tensor:
        """Sample n_samples synthetic embeddings for a given disease class."""
        self.eval()
        c = torch.zeros(n_samples, self.num_classes)
        c[:, disease_id] = 1.0
        z = torch.randn(n_samples, _LATENT_DIM)
        return self.decode(z, c)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, name: str = "cvae.pt") -> None:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / name
        torch.save(self.state_dict(), path)
        logger.info(f"VAE saved to {path}")

    def load(self, name: str = "cvae.pt") -> "ConditionalVAE":
        path = _OUTPUT_DIR / name
        self.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"VAE loaded from {path}")
        return self
