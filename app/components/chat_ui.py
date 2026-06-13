"""
chat_ui.py
Custom CSS injection and helper functions for the chat-style UI.
Call inject_styles() once at the top of the diagnosis page.
"""

import streamlit as st


def inject_styles() -> None:
    """Inject custom CSS for chat bubble styling."""
    st.markdown("""
    <style>
        /* User bubble */
        [data-testid="stChatMessageContent"][data-role="user"] {
            background-color: #EBF5FB;
            border-radius: 12px 12px 2px 12px;
            padding: 10px 14px;
        }
        /* Assistant bubble */
        [data-testid="stChatMessageContent"][data-role="assistant"] {
            background-color: #F9F9F9;
            border-left: 4px solid #2E75B6;
            border-radius: 2px 12px 12px 12px;
            padding: 10px 14px;
        }
        /* Disclaimer banner */
        .disclaimer-banner {
            background: #FFF3CD;
            border: 1px solid #F39C12;
            border-radius: 6px;
            padding: 8px 14px;
            font-size: 0.85em;
            color: #7D5A00;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)


def disclaimer_banner() -> None:
    st.markdown(
        '<div class="disclaimer-banner">'
        '⚠️ <strong>Educational simulation only.</strong> '
        'Not approved for clinical use. Always consult a licensed physician.'
        '</div>',
        unsafe_allow_html=True,
    )
