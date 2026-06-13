"""
augment.py
Phase 2 data augmentation utilities.
Primary path:  Conditional VAE (ConditionalVAE from src/models/vae.py).
Fallback path: SMOTE (imbalanced-learn) — used if VAE training is unstable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from src.models.vae import ConditionalVAE
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TARGET_SAMPLES  = get("vae.augmentation.target_samples_per_rare_class", 250)
_RARE_THRESHOLD  = get("data.rare_class_threshold", 100)
_LR              = get("vae.training.learning_rate", 1e-3)
_BATCH           = get("vae.training.batch_size", 32)
_EPOCHS          = get("vae.training.num_epochs", 50)
_KL_WEIGHT       = get("vae.training.kl_weight", 0.5)


def identify_rare_classes(df: pd.DataFrame) -> list[str]:
    """Return list of disease names whose sample count is below threshold."""
    counts = df["disease"].value_counts()
    rare   = counts[counts < _RARE_THRESHOLD].index.tolist()
    logger.info(f"Rare classes identified ({len(rare)}): {rare}")
    return rare


# ── VAE Training ──────────────────────────────────────────────────────────────

def train_vae(embeddings: np.ndarray, labels: np.ndarray,
              num_classes: int) -> ConditionalVAE:
    """
    Train the Conditional VAE on symptom embeddings.
    embeddings: (N, embed_dim) float32 array
    labels:     (N,) int array of disease class indices
    """
    input_dim = embeddings.shape[1]
    vae       = ConditionalVAE(input_dim=input_dim, num_classes=num_classes)
    optimiser = torch.optim.Adam(vae.parameters(), lr=_LR)

    X  = torch.tensor(embeddings, dtype=torch.float32)
    Y  = torch.zeros(len(labels), num_classes)
    Y[range(len(labels)), labels] = 1.0

    loader = DataLoader(TensorDataset(X, Y), batch_size=_BATCH, shuffle=True)

    vae.train()
    for epoch in range(1, _EPOCHS + 1):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            x_recon, mu, log_var = vae(x_batch, y_batch)
            loss = ConditionalVAE.loss(x_batch, x_recon, mu, log_var, _KL_WEIGHT)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"VAE Epoch {epoch:03d}/{_EPOCHS} | Loss: {epoch_loss/len(loader):.4f}")

    vae.save()
    return vae


def generate_synthetic_samples(vae: ConditionalVAE,
                                rare_class_ids: list[int],
                                id2label: dict[int, str],
                                existing_counts: dict[str, int]) -> pd.DataFrame:
    """
    Generate synthetic embedding vectors for rare classes until each
    reaches _TARGET_SAMPLES. Returns a DataFrame of synthetic records.
    Note: embeddings are saved as numpy arrays in a separate file;
    this DataFrame records disease labels and a synthetic flag.
    """
    rows = []
    for cls_id in rare_class_ids:
        label      = id2label[cls_id]
        n_existing = existing_counts.get(label, 0)
        n_needed   = max(0, _TARGET_SAMPLES - n_existing)
        if n_needed == 0:
            continue
        synthetic = vae.generate(disease_id=cls_id, n_samples=n_needed).numpy()
        for vec in synthetic:
            rows.append({"disease": label, "is_synthetic": True, "embedding": vec.tolist()})
        logger.info(f"Generated {n_needed} synthetic samples for '{label}'.")

    return pd.DataFrame(rows)


# ── SMOTE Fallback ────────────────────────────────────────────────────────────

def smote_augment(embeddings: np.ndarray,
                  labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback: apply SMOTE oversampling if VAE training is unstable.
    Returns augmented (embeddings, labels) arrays.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn not installed. Run: pip install imbalanced-learn")

    logger.info("Applying SMOTE oversampling (VAE fallback)...")
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(embeddings, labels)
    logger.info(f"SMOTE complete: {len(embeddings)} → {len(X_res)} samples.")
    return X_res, y_res


def validate_synthetic(real_embeddings: np.ndarray,
                        synthetic_embeddings: np.ndarray) -> dict:
    """
    Statistical validation: compare mean and variance of real vs synthetic
    embeddings per dimension. Returns summary statistics.
    """
    real_mean = np.mean(real_embeddings, axis=0)
    syn_mean  = np.mean(synthetic_embeddings, axis=0)
    real_std  = np.std(real_embeddings, axis=0)
    syn_std   = np.std(synthetic_embeddings, axis=0)
    mean_diff = np.mean(np.abs(real_mean - syn_mean))
    std_diff  = np.mean(np.abs(real_std  - syn_std))

    result = {
        "mean_absolute_diff": float(mean_diff),
        "std_absolute_diff":  float(std_diff),
        "pass": mean_diff < 0.5 and std_diff < 0.5,   # tunable threshold
    }
    logger.info(f"Synthetic validation: mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}, pass={result['pass']}")
    return result
