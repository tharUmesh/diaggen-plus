"""
diagnosis.py — Main diagnosis interaction page.
Gate 1: single-shot symptom input → top-3 predictions + LLM explanation.
Gate 3: iterative QA loop (activate by setting QA_LOOP_ENABLED = True).
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from app.components.chat_ui import inject_styles, disclaimer_banner
from app.components.confidence_bar import render_predictions, render_turn_indicator
from src.utils.config_loader import get

# ── Feature flag: flip to True once Phase 3 QA engine is stable ──────────────
QA_LOOP_ENABLED = False


# ── Model loader (cached across sessions) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading diagnostic model...")
def _load_classifier():
    from src.models.classifier import DiagnosisClassifier
    from src.training.trainer import build_label_maps

    processed_dir = Path("data/processed")
    train_df  = pd.read_csv(processed_dir / get("data.train_file", "train_imbalanced.csv"))
    label2id, id2label, _ = build_label_maps(train_df)
    clf = DiagnosisClassifier(label2id, id2label)
    return clf.load()


@st.cache_resource(show_spinner="Loading QA engine...")
def _load_qa_engine():
    from src.qa.engine import QAEngine
    clf = _load_classifier()
    return QAEngine(clf)


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":        [],
        "qa_state":        None,
        "awaiting_answer": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Page render ───────────────────────────────────────────────────────────────
def render():
    inject_styles()
    _init_state()

    st.header("🩺 Symptom Checker")
    st.caption(
        "Describe your symptoms in plain language. "
        + ("The system may ask follow-up questions to refine the diagnosis." if QA_LOOP_ENABLED
           else "The system will analyse them and provide a probable diagnosis.")
    )
    disclaimer_banner()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "predictions" in msg:
                render_predictions(msg["predictions"])
            if "turn_info" in msg:
                render_turn_indicator(*msg["turn_info"])

    # Input
    placeholder = (
        "Answer the question above..."
        if st.session_state.awaiting_answer
        else "e.g. 'I have had fever, joint pain, and a skin rash for 5 days'"
    )

    if user_input := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                response_data = _process(user_input)
            st.markdown(response_data["text"])
            if response_data.get("predictions"):
                render_predictions(response_data["predictions"])
            if response_data.get("turn_info"):
                render_turn_indicator(*response_data["turn_info"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_data["text"],
            "predictions": response_data.get("predictions"),
            "turn_info": response_data.get("turn_info"),
        })

    # Reset
    if st.session_state.messages:
        if st.button("🔄 Start new session"):
            st.session_state.messages        = []
            st.session_state.qa_state         = None
            st.session_state.awaiting_answer  = False
            st.rerun()


# ── Processing logic ──────────────────────────────────────────────────────────
def _process(user_input: str) -> dict:
    if QA_LOOP_ENABLED:
        return _process_qa_loop(user_input)
    return _process_single_shot(user_input)


def _process_single_shot(user_input: str) -> dict:
    """Gate 1 path — classify once, generate explanation, return."""
    from src.preprocessing.cleaner import preprocess
    from src.qa.llm_client import call, format_prompt

    try:
        clf    = _load_classifier()
        clean  = preprocess(user_input)
        result = clf.predict(clean)
        preds  = result["predictions"]

        preds_str = "\n".join(
            f"{i+1}. {p['disease']} — {p['confidence']:.1%}"
            for i, p in enumerate(preds)
        )
        prompt = format_prompt(
            "explanation.template",
            symptoms=user_input,
            predictions=preds_str,
            top_disease=result["top_disease"],
        )
        explanation = call(prompt)

        return {
            "text": explanation,
            "predictions": preds,
        }

    except FileNotFoundError:
        return {
            "text": (
                "⚠️ **Model not yet trained.**\n\n"
                "Run `notebooks/phase1_bert_finetuning.ipynb` first to train the classifier, "
                "then restart the app."
            ),
            "predictions": None,
        }
    except Exception as e:
        return {
            "text": f"⚠️ An error occurred: `{e}`\n\nCheck the terminal for details.",
            "predictions": None,
        }


def _process_qa_loop(user_input: str) -> dict:
    """Gate 3 path — iterative QA loop."""
    from src.qa.engine import QAEngine
    max_turns = get("qa.max_turns", 4)

    engine = _load_qa_engine()

    if st.session_state.qa_state is None:
        # First message — start session
        state = engine.start(user_input)
        st.session_state.qa_state = state
    else:
        # Follow-up answer
        state = engine.advance(st.session_state.qa_state, user_input)
        st.session_state.qa_state = state

    if state.terminated:
        st.session_state.awaiting_answer = False
        explanation = engine.explain(state)
        return {
            "text": explanation,
            "predictions": state.final_predictions,
        }
    else:
        st.session_state.awaiting_answer = True
        return {
            "text": f"**Follow-up question:**\n\n{state.next_question}",
            "predictions": state.final_predictions,
            "turn_info": (state.turn_count, max_turns),
        }
