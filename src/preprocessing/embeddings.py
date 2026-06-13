"""
embeddings.py
Loads and manages GloVe / FastText / Word2Vec embeddings.
Used for EDA, feature analysis, and as input to non-BERT models.
(BERT has its own internal embeddings — this module is for the NLP layer.)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import gensim.downloader as gensim_api
from gensim.models import FastText, KeyedVectors
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TYPE      = get("nlp.embeddings.type", "glove")
_GLOVE_DIM = get("nlp.embeddings.glove_dim", 100)
_FT_DIM    = get("nlp.embeddings.fasttext_dim", 300)

# Gensim pretrained model identifiers
_GENSIM_GLOVE_MAP = {
    50:  "glove-wiki-gigaword-50",
    100: "glove-wiki-gigaword-100",
    200: "glove-wiki-gigaword-200",
    300: "glove-wiki-gigaword-300",
}


class EmbeddingModel:
    """Thin wrapper around a loaded KeyedVectors / FastText model."""

    def __init__(self):
        self._model: KeyedVectors | None = None

    def load(self) -> "EmbeddingModel":
        if _TYPE == "glove":
            name = _GENSIM_GLOVE_MAP.get(_GLOVE_DIM, "glove-wiki-gigaword-100")
            logger.info(f"Loading GloVe model: {name} (this may take a moment on first run)")
            self._model = gensim_api.load(name)
        elif _TYPE == "fasttext":
            logger.info("Loading FastText model: cc.en.300.bin")
            self._model = gensim_api.load("fasttext-wiki-news-subwords-300")
        else:
            raise ValueError(f"Unsupported embedding type: {_TYPE}. Choose 'glove' or 'fasttext'.")
        logger.info("Embeddings loaded.")
        return self

    def embed_word(self, word: str) -> np.ndarray | None:
        """Return vector for a single word, or None if OOV."""
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        try:
            return self._model[word]
        except KeyError:
            return None

    def embed_sentence(self, tokens: list[str]) -> np.ndarray:
        """
        Mean-pool token vectors into a single sentence vector.
        OOV tokens are ignored. Returns zero vector if all tokens are OOV.
        """
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        vecs = [self._model[t] for t in tokens if t in self._model]
        if not vecs:
            dim = self._model.vector_size
            return np.zeros(dim)
        return np.mean(vecs, axis=0)

    @property
    def dim(self) -> int:
        if self._model is None:
            raise RuntimeError("Call .load() first.")
        return self._model.vector_size


# Module-level singleton — call embeddings.load() once in notebooks/scripts
embeddings = EmbeddingModel()
