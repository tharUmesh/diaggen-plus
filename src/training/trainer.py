"""
trainer.py
Higher-level training orchestration.
Handles data loading, train/val/test splitting, and invokes
the BERT Trainer and VAE training loop.
Designed to be called from notebooks or CLI.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.preprocessing.cleaner import preprocess_batch
from src.models.classifier import DiagnosisClassifier
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TRAIN_RATIO    = get("data.split.train", 0.80)
_VAL_RATIO      = get("data.split.val",   0.10)
_SEED           = get("data.split.random_seed", 42)
_STRATIFY       = get("data.split.stratify", True)

# Resolve paths relative to project root (parent of src/)
_PROJECT_ROOT   = Path(__file__).parent.parent.parent
_PROCESSED_DIR  = _PROJECT_ROOT / get("data.processed_dir", "data/processed")
_TRAIN_FILE     = get("data.train_file", "train_imbalanced.csv")
_VAL_FILE       = get("data.val_file",   "val_imbalanced.csv")
_TEST_FILE      = get("data.test_file",  "test_imbalanced.csv")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset. Expects 'symptoms' and 'disease' columns."""
    df = pd.read_csv(path)
    required = {"symptoms", "disease"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")
    logger.info(f"Loaded dataset: {len(df)} rows, {df['disease'].nunique()} disease classes.")
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 train/val/test split."""
    stratify = df["disease"] if _STRATIFY else None
    test_size = 1 - _TRAIN_RATIO

    train_df, temp_df = train_test_split(df, test_size=test_size,
                                         random_state=_SEED, stratify=stratify)

    val_ratio_adjusted = _VAL_RATIO / (1 - _TRAIN_RATIO)
    stratify_temp = temp_df["disease"] if _STRATIFY else None
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_ratio_adjusted,
                                       random_state=_SEED, stratify=stratify_temp)

    logger.info(f"Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_label_maps(df: pd.DataFrame) -> tuple[dict, dict, LabelEncoder]:
    """Encode disease labels. Returns (label2id, id2label, encoder)."""
    le       = LabelEncoder()
    le.fit(df["disease"])
    classes  = list(le.classes_)
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}
    logger.info(f"Label map built: {len(classes)} classes.")
    return label2id, id2label, le


def load_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-split CSVs written by phase0_eda.ipynb.
    Use this in Phase 1 onwards instead of re-splitting from scratch.
    """
    train_df = pd.read_csv(_PROCESSED_DIR / _TRAIN_FILE)
    val_df   = pd.read_csv(_PROCESSED_DIR / _VAL_FILE)
    test_df  = pd.read_csv(_PROCESSED_DIR / _TEST_FILE)
    logger.info(f"Loaded splits — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def run_bert_training(data_path: str | Path) -> DiagnosisClassifier:
    """
    Full Phase 1 training pipeline.
    Loads data → splits → preprocesses → fine-tunes BERT → returns classifier.
    """
    df = load_dataset(data_path)
    train_df, val_df, test_df = split_data(df)
    label2id, id2label, le    = build_label_maps(df)

    # Preprocess text
    logger.info("Preprocessing symptom text (this may take a moment)...")
    train_texts = preprocess_batch(train_df["symptoms"].tolist())
    val_texts   = preprocess_batch(val_df["symptoms"].tolist())

    train_labels = le.transform(train_df["disease"].tolist()).tolist()
    val_labels   = le.transform(val_df["disease"].tolist()).tolist()

    # Train
    clf       = DiagnosisClassifier(label2id, id2label)
    train_ds  = clf.prepare_dataset(train_texts, train_labels)
    val_ds    = clf.prepare_dataset(val_texts, val_labels)
    clf.train(train_ds, val_ds)

    # Return classifier + save test set for evaluation notebook
    test_path = Path(get("data.processed_dir", "data/processed")) / "test_set.csv"
    test_df.to_csv(test_path, index=False)
    logger.info(f"Test set saved to {test_path} for evaluation.")

    return clf
