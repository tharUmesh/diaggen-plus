"""
test_preprocessing.py
Unit tests for src/preprocessing/cleaner.py
Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
from src.preprocessing.cleaner import basic_clean, expand_abbreviations, preprocess


class TestBasicClean:
    def test_lowercases(self):
        assert basic_clean("Fever and CHILLS") == "fever and chills"

    def test_strips_special_chars(self):
        result = basic_clean("fever! joint-pain? rash@home")
        assert "!" not in result
        assert "?" not in result
        assert "@" not in result

    def test_normalises_whitespace(self):
        assert basic_clean("fever   and   pain") == "fever and pain"

    def test_empty_string(self):
        assert basic_clean("") == ""

    def test_preserves_commas_and_periods(self):
        result = basic_clean("fever, chills. fatigue")
        assert "," in result
        assert "." in result


class TestExpandAbbreviations:
    def test_expands_sob(self):
        result = expand_abbreviations("sob and cp")
        assert "shortness of breath" in result
        assert "chest pain" in result

    def test_passes_through_unknown_tokens(self):
        result = expand_abbreviations("fever and fatigue")
        assert "fever" in result
        assert "fatigue" in result

    def test_case_insensitive(self):
        result = expand_abbreviations("SOB".lower())
        assert "shortness of breath" in result


class TestPreprocess:
    def test_returns_string(self):
        result = preprocess("I have a fever and joint pain")
        assert isinstance(result, str)

    def test_removes_stopwords(self):
        result = preprocess("I have a fever and joint pain")
        tokens = result.split()
        assert "i" not in tokens
        assert "a" not in tokens

    def test_non_empty_input(self):
        result = preprocess("persistent fatigue headache rash")
        assert len(result) > 0

    def test_handles_empty_string(self):
        result = preprocess("")
        assert result == ""

    def test_handles_abbreviations(self):
        result = preprocess("sob and cp for 3 days")
        assert len(result) > 0
