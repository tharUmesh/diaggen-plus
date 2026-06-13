"""
diagnosis.py — Main diagnosis interaction page.
Phase 1 (Gate 1): Single-shot symptom input → top-3 predictions + explanation.
Phase 3 (Gate 3): Iterative QA loop — will be activated when QAEngine is ready.
"""

import streamlit as st

# ── Page-level session state initialisation ────────────────────────────────────
def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages     = []
    if "qa_state" not in st.session_state:
        st.session_state.qa_state      = None
    if "awaiting_answer" not in st.session_state:
        st.session_state.awaiting_answer = False


def render():
    _init_state()
    st.header("🩺 Symptom Checker")
    st.caption("Describe your symptoms in plain language. The system will analyse them and may ask follow-up questions.")

    # ── Display chat history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    placeholder = (
        "Answer the question above..."
        if st.session_state.awaiting_answer
        else "Describe your symptoms (e.g. 'I have had fever, joint pain, and a rash for 5 days')"
    )

    if user_input := st.chat_input(placeholder):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing symptoms..."):
                response = _process(user_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # ── Reset button ──────────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🔄 Start new session"):
            st.session_state.messages        = []
            st.session_state.qa_state         = None
            st.session_state.awaiting_answer  = False
            st.rerun()


def _process(user_input: str) -> str:
    """
    Route user input to the appropriate handler.
    Phase 1: classifier only.
    Phase 3: QA engine (uncomment when ready).
    """
    # ── Phase 1 path (always available as fallback) ───────────────────────────
    # TODO Phase 1: load classifier, run predict(), call llm_client for explanation
    # Placeholder response until Phase 1 model is trained:
    return (
        "**[PLACEHOLDER — Phase 1 not yet implemented]**\n\n"
        f"You entered: *{user_input}*\n\n"
        "Once the BERT model is trained (Phase 1), this will display:\n"
        "- Top-3 disease predictions with confidence scores\n"
        "- A chain-of-thought explanation\n\n"
        "> ⚠️ EDUCATIONAL SIMULATION ONLY. Not approved for clinical use."
    )

    # ── Phase 3 path (uncomment when QA engine is ready) ─────────────────────
    # from src.qa.engine import QAEngine
    # from src.models.classifier import DiagnosisClassifier
    # ...
