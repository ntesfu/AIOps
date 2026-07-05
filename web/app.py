from __future__ import annotations

import tempfile
import json
from pathlib import Path

import streamlit as st

from aiops.inference.pipeline import run_baseline_inference
from aiops.procedure import Procedure


st.set_page_config(page_title="AIOps", layout="wide")

st.title("AIOps")

procedure_path = st.text_input("Procedure JSON", value="configs/example_procedure.json")
uploaded_video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video and st.button("Analyze", type="primary"):
    procedure = Procedure.from_json(procedure_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as handle:
        handle.write(uploaded_video.getbuffer())
        video_path = handle.name

    payload = run_baseline_inference(video_path, procedure)
    st.subheader("Predicted Segments")
    st.dataframe(payload["predictions"], use_container_width=True)
    st.subheader("Validation Events")
    st.dataframe(payload["validation_events"], use_container_width=True)
    st.download_button(
        "Download JSON",
        data=json.dumps(payload, indent=2),
        file_name="aiops_predictions.json",
        mime="application/json",
    )
