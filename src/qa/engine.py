"""
engine.py
The Medical QA Loop — conversation state manager.
Orchestrates the iterative symptom-refinement loop between the
BERT classifier, follow-up question generator, and final explanation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from src.utils.config_loader import get
from src.utils.logger import get_logger
from src.qa import llm_client

logger = get_logger(__name__)

_MAX_TURNS  = get("qa.max_turns", 4)
_THRESHOLD  = get("qa.confidence_threshold", 0.80)
_MIN_DELTA  = get("qa.min_confidence_delta", 0.05)


@dataclass
class ConversationState:
    """Holds the full context of a diagnostic session."""
    initial_symptoms:   str
    turns:              list[dict] = field(default_factory=list)   # {"question": str, "answer": str}
    predictions_history: list[list[dict]] = field(default_factory=list)
    final_predictions:  list[dict] | None = None
    terminated:         bool = False
    termination_reason: str  = ""   # "threshold_reached" | "max_turns" | "no_delta"

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def full_context(self) -> str:
        """Build the concatenated context string for re-classification."""
        parts = [self.initial_symptoms]
        for t in self.turns:
            parts.append(f"Q: {t['question']} A: {t['answer']}")
        return " ".join(parts)

    @property
    def conversation_history_str(self) -> str:
        if not self.turns:
            return "(No follow-up questions asked yet.)"
        lines = []
        for i, t in enumerate(self.turns, 1):
            lines.append(f"Turn {i}: [{t['question']}] → [{t['answer']}]")
        return "\n".join(lines)


class QAEngine:
    """
    Orchestrates the full diagnostic session.

    Usage:
        engine = QAEngine(classifier)
        state  = engine.start("I have fever and joint pain")
        # If state.terminated: go straight to explain()
        # Else: show state.next_question to the user, collect answer, then:
        state  = engine.advance(state, user_answer)
        # Repeat until state.terminated is True
        result = engine.explain(state)
    """

    def __init__(self, classifier):
        """classifier: an instance of DiagnosisClassifier with a loaded model."""
        self.classifier = classifier

    def start(self, symptoms: str) -> ConversationState:
        """Initialise a session and run the first classification pass."""
        state = ConversationState(initial_symptoms=symptoms)
        preds = self.classifier.predict(symptoms)
        state.predictions_history.append(preds["predictions"])
        state.final_predictions = preds["predictions"]

        if preds["above_threshold"]:
            state.terminated       = True
            state.termination_reason = "threshold_reached"
            logger.info(f"Confident on first pass: {preds['top_disease']} ({preds['top_confidence']:.0%})")
        else:
            state.next_question = self._generate_question(state, preds["predictions"])
            logger.info(f"Confidence {preds['top_confidence']:.0%} below threshold — asking follow-up.")

        return state

    def advance(self, state: ConversationState, user_answer: str) -> ConversationState:
        """Record user answer, re-classify, decide whether to continue or terminate."""
        if state.terminated:
            return state

        # Record this turn
        state.turns.append({"question": state.next_question, "answer": user_answer})

        # Re-classify with updated context
        preds      = self.classifier.predict(state.full_context)
        prev_conf  = state.final_predictions[0]["confidence"] if state.final_predictions else 0.0
        curr_conf  = preds["predictions"][0]["confidence"]
        state.predictions_history.append(preds["predictions"])
        state.final_predictions = preds["predictions"]

        # Termination checks
        if preds["above_threshold"]:
            state.terminated       = True
            state.termination_reason = "threshold_reached"
        elif state.turn_count >= _MAX_TURNS:
            state.terminated       = True
            state.termination_reason = "max_turns"
        elif abs(curr_conf - prev_conf) < _MIN_DELTA:
            state.terminated       = True
            state.termination_reason = "no_delta"
        else:
            state.next_question = self._generate_question(state, preds["predictions"])

        logger.info(f"Turn {state.turn_count} | conf={curr_conf:.0%} | terminated={state.terminated}")
        return state

    def explain(self, state: ConversationState) -> str:
        """Generate the final doctor-like chain-of-thought explanation."""
        preds_str = self._format_predictions(state.final_predictions)

        if state.termination_reason == "threshold_reached" and state.turn_count == 0:
            # Confident on first pass — single-shot explanation
            prompt = llm_client.format_prompt(
                "explanation.template",
                symptoms=state.initial_symptoms,
                predictions=preds_str,
                top_disease=state.final_predictions[0]["disease"],
            )
        elif state.termination_reason in ("threshold_reached", "max_turns", "no_delta") \
                and state.turn_count > 0:
            # Post-QA explanation
            prompt = llm_client.format_prompt(
                "qa_final_explanation.template",
                initial_symptoms=state.initial_symptoms,
                conversation_history=state.conversation_history_str,
                final_predictions=preds_str,
                top_disease=state.final_predictions[0]["disease"],
            )
        else:
            # Low-confidence fallback
            prompt = llm_client.format_prompt(
                "low_confidence_explanation.template",
                symptoms=state.initial_symptoms,
                num_turns=state.turn_count,
                predictions=preds_str,
            )

        return llm_client.call(prompt)

    # ── Private helpers ────────────────────────────────────────────────────────
    def _generate_question(self, state: ConversationState, predictions: list[dict]) -> str:
        preds_str = self._format_predictions(predictions)
        prompt = llm_client.format_prompt(
            "followup_question.template",
            symptoms=state.full_context,
            predictions=preds_str,
            top_confidence=predictions[0]["confidence"],
            threshold=_THRESHOLD,
            conversation_history=state.conversation_history_str,
        )
        return llm_client.call(prompt)

    @staticmethod
    def _format_predictions(preds: list[dict]) -> str:
        return "\n".join(
            f"{i+1}. {p['disease']} — {p['confidence']:.1%}"
            for i, p in enumerate(preds)
        )
