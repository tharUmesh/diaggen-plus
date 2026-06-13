"""
test_classifier.py
Unit tests for src/models/classifier.py
Run with: pytest tests/test_classifier.py -v
"""

import pytest
import numpy as np


class TestDiagnosisClassifier:
    """Tests that can run without a trained model checkpoint."""

    @pytest.fixture
    def label_maps(self):
        diseases  = ["Flu", "Malaria", "Dengue", "Typhoid", "Lupus"]
        label2id  = {d: i for i, d in enumerate(diseases)}
        id2label  = {i: d for i, d in enumerate(diseases)}
        return label2id, id2label

    def test_classifier_instantiation(self, label_maps):
        from src.models.classifier import DiagnosisClassifier
        label2id, id2label = label_maps
        clf = DiagnosisClassifier(label2id, id2label)
        assert clf.num_labels == 5
        assert clf.label2id   == label2id
        assert clf.id2label   == id2label

    def test_tokenizer_loads(self, label_maps):
        from src.models.classifier import DiagnosisClassifier
        label2id, id2label = label_maps
        clf = DiagnosisClassifier(label2id, id2label)
        # Tokenizer should be available after init
        assert clf.tokenizer is not None

    def test_prepare_dataset_structure(self, label_maps):
        from src.models.classifier import DiagnosisClassifier
        label2id, id2label = label_maps
        clf    = DiagnosisClassifier(label2id, id2label)
        texts  = ["fever and chills", "joint pain rash fatigue"]
        labels = [0, 4]
        ds     = clf.prepare_dataset(texts, labels)
        assert "input_ids"      in ds.column_names
        assert "attention_mask" in ds.column_names
        assert "label"          in ds.column_names
        assert len(ds) == 2

    def test_predict_raises_without_model(self, label_maps):
        from src.models.classifier import DiagnosisClassifier
        label2id, id2label = label_maps
        clf = DiagnosisClassifier(label2id, id2label)
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict("fever and joint pain")

    def test_predict_output_structure(self, label_maps, monkeypatch):
        """Mock the model to test predict() output format without a real checkpoint."""
        import torch
        from src.models.classifier import DiagnosisClassifier

        label2id, id2label = label_maps
        clf = DiagnosisClassifier(label2id, id2label)

        # Inject a mock model
        class _MockOutput:
            logits = torch.tensor([[2.0, 0.5, 0.3, 0.1, 0.05]])

        class _MockModel:
            def __call__(self, **kwargs): return _MockOutput()
            def eval(self): pass

        clf.model = _MockModel()
        result = clf.predict("fever and chills")

        assert "top_disease"     in result
        assert "top_confidence"  in result
        assert "above_threshold" in result
        assert "predictions"     in result
        assert len(result["predictions"]) <= 3
        assert result["top_disease"] in id2label.values()
        assert 0.0 <= result["top_confidence"] <= 1.0
