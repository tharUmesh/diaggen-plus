"""
confidence_bar.py
Reusable Streamlit component: renders a colour-coded confidence bar
and top-K prediction cards for a list of disease predictions.
"""

import streamlit as st


_COLOUR_HIGH   = "#27AE60"   # green  — above threshold
_COLOUR_MED    = "#F39C12"   # amber  — borderline
_COLOUR_LOW    = "#E74C3C"   # red    — low confidence
_THRESHOLD     = 0.80


def _confidence_colour(conf: float) -> str:
    if conf >= _THRESHOLD:
        return _COLOUR_HIGH
    elif conf >= 0.50:
        return _COLOUR_MED
    return _COLOUR_LOW


def render_predictions(predictions: list[dict]) -> None:
    """
    Renders top-K prediction cards.
    Each prediction dict: {"disease": str, "confidence": float}
    """
    if not predictions:
        return

    st.markdown("#### 🔍 Top Predictions")
    for i, pred in enumerate(predictions):
        conf  = pred["confidence"]
        col   = _confidence_colour(conf)
        label = "✅ Primary" if i == 0 else f"#{i+1}"

        st.markdown(
            f"""
            <div style="
                border-left: 5px solid {col};
                background: {'#f0fff4' if i == 0 else '#fafafa'};
                padding: 10px 16px;
                margin-bottom: 8px;
                border-radius: 4px;
            ">
                <span style="color:{col}; font-weight:700; font-size:0.85em;">{label}</span>
                <span style="font-size:1.05em; font-weight:600; margin-left:8px;">{pred['disease']}</span>
                <div style="background:#e0e0e0; border-radius:4px; margin-top:6px; height:10px;">
                    <div style="width:{conf*100:.1f}%; background:{col}; height:10px; border-radius:4px;"></div>
                </div>
                <span style="font-size:0.82em; color:#555;">{conf:.1%} confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_turn_indicator(current_turn: int, max_turns: int) -> None:
    """Shows how many follow-up rounds have been used."""
    st.markdown(
        f"**Follow-up round:** {current_turn} / {max_turns}",
        help="The system will commit to a diagnosis after the maximum number of follow-up questions.",
    )
    st.progress(current_turn / max_turns)
