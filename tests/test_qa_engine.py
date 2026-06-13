"""
test_qa_engine.py
Unit tests for src/qa/engine.py
Run with: pytest tests/test_qa_engine.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from src.qa.engine import ConversationState, QAEngine


# ── ConversationState tests ───────────────────────────────────────────────────

class TestConversationState:
    def test_initial_state(self):
        state = ConversationState(initial_symptoms="fever and joint pain")
        assert state.initial_symptoms == "fever and joint pain"
        assert state.turn_count        == 0
        assert state.terminated        == False
        assert state.turns             == []

    def test_full_context_no_turns(self):
        state = ConversationState(initial_symptoms="fever")
        assert state.full_context == "fever"

    def test_full_context_with_turns(self):
        state = ConversationState(initial_symptoms="fever")
        state.turns.append({"question": "Do you have a rash?", "answer": "Yes"})
        assert "fever" in state.full_context
        assert "Do you have a rash?" in state.full_context
        assert "Yes" in state.full_context

    def test_turn_count(self):
        state = ConversationState(initial_symptoms="fever")
        assert state.turn_count == 0
        state.turns.append({"question": "Q1", "answer": "A1"})
        assert state.turn_count == 1
        state.turns.append({"question": "Q2", "answer": "A2"})
        assert state.turn_count == 2

    def test_conversation_history_str_empty(self):
        state = ConversationState(initial_symptoms="fever")
        assert "No follow-up" in state.conversation_history_str

    def test_conversation_history_str_with_turns(self):
        state = ConversationState(initial_symptoms="fever")
        state.turns.append({"question": "Do you have chills?", "answer": "Yes, severe"})
        hist = state.conversation_history_str
        assert "Turn 1" in hist
        assert "Do you have chills?" in hist
        assert "Yes, severe" in hist


# ── QAEngine tests ─────────────────────────────────────────────────────────────

class TestQAEngine:
    @pytest.fixture
    def mock_classifier(self):
        clf = MagicMock()
        clf.predict.return_value = {
            "top_disease":    "Flu",
            "top_confidence": 0.65,
            "above_threshold": False,
            "predictions": [
                {"disease": "Flu",    "confidence": 0.65},
                {"disease": "Dengue", "confidence": 0.20},
                {"disease": "Typhoid","confidence": 0.15},
            ],
        }
        return clf

    @pytest.fixture
    def confident_classifier(self):
        clf = MagicMock()
        clf.predict.return_value = {
            "top_disease":    "Malaria",
            "top_confidence": 0.92,
            "above_threshold": True,
            "predictions": [{"disease": "Malaria", "confidence": 0.92}],
        }
        return clf

    def test_engine_terminates_immediately_on_high_confidence(self, confident_classifier):
        with patch("src.qa.engine.llm_client"):
            engine = QAEngine(confident_classifier)
            state  = engine.start("fever chills sweating")
        assert state.terminated == True
        assert state.termination_reason == "threshold_reached"
        assert state.turn_count == 0

    def test_engine_generates_question_on_low_confidence(self, mock_classifier):
        with patch("src.qa.engine.llm_client") as mock_llm:
            mock_llm.format_prompt.return_value = "formatted prompt"
            mock_llm.call.return_value          = "Do you have a rash?"
            engine = QAEngine(mock_classifier)
            state  = engine.start("fever joint pain")
        assert state.terminated  == False
        assert state.next_question == "Do you have a rash?"

    def test_advance_records_turn(self, mock_classifier):
        with patch("src.qa.engine.llm_client") as mock_llm:
            mock_llm.format_prompt.return_value = "prompt"
            mock_llm.call.return_value          = "Next question?"
            engine = QAEngine(mock_classifier)
            state  = engine.start("fever")
            state  = engine.advance(state, "Yes, I have a rash")
        assert state.turn_count == 1
        assert state.turns[0]["answer"] == "Yes, I have a rash"

    def test_advance_terminates_at_max_turns(self, mock_classifier):
        with patch("src.qa.engine.llm_client") as mock_llm:
            mock_llm.format_prompt.return_value = "prompt"
            mock_llm.call.return_value          = "Another question?"
            engine = QAEngine(mock_classifier)
            state  = engine.start("fever")
            # Exhaust all turns
            for _ in range(4):   # _MAX_TURNS = 4
                if not state.terminated:
                    state = engine.advance(state, "yes")
        assert state.terminated == True
        assert state.termination_reason in ("max_turns", "threshold_reached", "no_delta")
