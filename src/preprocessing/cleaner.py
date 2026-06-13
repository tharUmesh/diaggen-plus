"""
cleaner.py
Text preprocessing pipeline for symptom descriptions.
Handles tokenisation, stop-word removal, and lemmatisation via spaCy.
"""

from __future__ import annotations
import re
import spacy
from src.utils.config_loader import get
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME   = get("nlp.spacy_model", "en_core_web_sm")
_RM_STOPWORDS = get("nlp.remove_stopwords", True)
_LEMMATIZE    = get("nlp.lemmatize", True)

# Medical abbreviation expansions (extend as needed)
_ABBREV = {
    "sob":  "shortness of breath",
    "cp":   "chest pain",
    "ha":   "headache",
    "gi":   "gastrointestinal",
    "n/v":  "nausea vomiting",
    "htn":  "hypertension",
    "dm":   "diabetes mellitus",
}


def _load_spacy() -> spacy.Language:
    try:
        return spacy.load(_MODEL_NAME)
    except OSError:
        logger.warning(f"spaCy model '{_MODEL_NAME}' not found. Run: python -m spacy download {_MODEL_NAME}")
        raise


_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy()
    return _nlp


def expand_abbreviations(text: str) -> str:
    """Replace known medical abbreviations with full forms."""
    tokens = text.lower().split()
    return " ".join(_ABBREV.get(t, t) for t in tokens)


def basic_clean(text: str) -> str:
    """Lowercase, strip special chars, normalise whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s,.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Basic cleaning
      2. Abbreviation expansion
      3. spaCy tokenisation + optional stop-word removal + optional lemmatisation
    Returns a clean string ready for embedding or BERT tokenisation.
    """
    text = basic_clean(text)
    text = expand_abbreviations(text)

    nlp  = _get_nlp()
    doc  = nlp(text)

    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if _RM_STOPWORDS and token.is_stop:
            continue
        word = token.lemma_ if _LEMMATIZE else token.text
        if word.strip():
            tokens.append(word.strip())

    return " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Vectorised batch preprocessing using spaCy's pipe for speed."""
    nlp = _get_nlp()
    results = []
    for doc in nlp.pipe([basic_clean(expand_abbreviations(t)) for t in texts],
                        batch_size=64):
        tokens = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if _RM_STOPWORDS and token.is_stop:
                continue
            word = token.lemma_ if _LEMMATIZE else token.text
            if word.strip():
                tokens.append(word.strip())
        results.append(" ".join(tokens))
    return results
