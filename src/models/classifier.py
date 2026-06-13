"""
classifier.py
Fine-tuned BERT/BioBERT classifier for symptom-to-disease prediction.
Wraps Hugging Face Trainer API with project-specific defaults.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_MODEL   = get("bert.base_model", "dmis-lab/biobert-base-cased-v1.2")
_MAX_LEN      = get("bert.max_length", 256)
_OUTPUT_DIR   = Path(get("bert.output_dir", "models/bert"))
_NUM_LABELS   = get("bert.num_labels", 24)
_THRESHOLD    = get("bert.confidence_threshold", 0.80)
_TOP_K        = get("bert.top_k_predictions", 3)

_LR           = get("bert.training.learning_rate", 2e-5)
_BATCH        = get("bert.training.batch_size", 16)
_EPOCHS       = get("bert.training.num_epochs", 5)
_WARMUP       = get("bert.training.warmup_ratio", 0.1)
_WD           = get("bert.training.weight_decay", 0.01)
_ES_PATIENCE  = get("bert.training.early_stopping_patience", 2)
_FP16         = get("bert.training.fp16", False)


class DiagnosisClassifier:
    """
    Wrapper for a fine-tuned BERT sequence classification model.
    Usage:
        clf = DiagnosisClassifier(label2id, id2label)
        clf.train(train_dataset, val_dataset)
        results = clf.predict("I have a fever and joint pain")
    """

    def __init__(self, label2id: dict[str, int], id2label: dict[int, str]):
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)
        self.tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        self.model: AutoModelForSequenceClassification | None = None

    # ── Tokenisation ──────────────────────────────────────────────────────────
    def tokenize(self, examples: dict) -> dict:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=_MAX_LEN,
        )

    def prepare_dataset(self, texts: list[str], labels: list[int]) -> Dataset:
        ds = Dataset.from_dict({"text": texts, "label": labels})
        return ds.map(self.tokenize, batched=True)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, train_ds: Dataset, val_ds: Dataset) -> None:
        logger.info(f"Loading base model: {_BASE_MODEL}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            _BASE_MODEL,
            num_labels=self.num_labels,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        )

        args = TrainingArguments(
            output_dir=str(_OUTPUT_DIR),
            learning_rate=_LR,
            per_device_train_batch_size=_BATCH,
            per_device_eval_batch_size=_BATCH,
            num_train_epochs=_EPOCHS,
            warmup_ratio=_WARMUP,
            weight_decay=_WD,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=_FP16,
            logging_dir=str(_OUTPUT_DIR / "logs"),
            logging_steps=50,
            report_to="none",              # set to "mlflow" when tracking enabled
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=_ES_PATIENCE)],
        )

        logger.info("Starting BERT fine-tuning...")
        trainer.train()
        logger.info(f"Training complete. Model saved to {_OUTPUT_DIR}")
        self.model.save_pretrained(_OUTPUT_DIR)
        self.tokenizer.save_pretrained(_OUTPUT_DIR)

    # ── Inference ─────────────────────────────────────────────────────────────
    def load(self, path: str | Path | None = None) -> "DiagnosisClassifier":
        """Load a saved model from disk."""
        load_path = Path(path) if path else _OUTPUT_DIR
        logger.info(f"Loading model from {load_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.eval()
        return self

    def predict(self, text: str) -> dict:
        """
        Returns:
            {
              "top_disease": str,
              "top_confidence": float,
              "above_threshold": bool,
              "predictions": [{"disease": str, "confidence": float}, ...]   # top-K
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .train() or .load() first.")

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=_MAX_LEN
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1).squeeze().numpy()
        top_idx = np.argsort(probs)[::-1][:_TOP_K]

        preds = [
            {"disease": self.id2label[i], "confidence": float(probs[i])}
            for i in top_idx
        ]
        return {
            "top_disease":       preds[0]["disease"],
            "top_confidence":    preds[0]["confidence"],
            "above_threshold":   preds[0]["confidence"] >= _THRESHOLD,
            "predictions":       preds,
        }
