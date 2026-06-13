"""
main.py — DiagGen+ Streamlit Entry Point
Run with: streamlit run app/main.py
"""

import streamlit as st
from src.utils.config_loader import get

_TITLE      = get("app.title",    "DiagGen+")
_SUBTITLE   = get("app.subtitle", "Medical QA Diagnostic Assistant")
_DISCLAIMER = get("app.disclaimer", "EDUCATIONAL SIMULATION ONLY.")
_COLOR      = get("app.theme_color", "#1F3864")

st.set_page_config(
    page_title=_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stethoscope.png", width=64)
    st.title(_TITLE)
    st.caption(_SUBTITLE)
    st.divider()
    page = st.radio("Navigation", ["Diagnosis", "About"], label_visibility="collapsed")
    st.divider()
    st.error(f"⚠️ {_DISCLAIMER}", icon="🚨")

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Diagnosis":
    from app.pages import diagnosis
    diagnosis.render()
else:
    from app.pages import about
    about.render()
