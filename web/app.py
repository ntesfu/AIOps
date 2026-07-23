from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Hand Atlas - Dataset Labeler",
    page_icon="HA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #eeece5; }
      header[data-testid="stHeader"], footer { display: none; }
      .block-container { padding: 0 !important; max-width: none !important; }
      iframe { display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)


APP_HTML = Path(__file__).with_name("labeler.html").read_text(encoding="utf-8")
components.html(APP_HTML, height=1260, scrolling=True)
