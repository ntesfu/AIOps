from __future__ import annotations

import tempfile
import json
from pathlib import Path

import streamlit as st

from aiops.architecture import AIOpsArchitectureConfig
from aiops.inference.pipeline import run_baseline_inference, run_temporal_inference
from aiops.procedure import Procedure


st.set_page_config(page_title="AIOps", layout="wide")

st.title("AIOps")

procedure_path = st.text_input("Procedure JSON", value="configs/example_procedure.json")
mode = st.radio("Mode", options=["Temporal", "Baseline"], horizontal=True)
architecture_path = st.text_input("Architecture JSON", value="configs/temporal_architecture.json")
checkpoint_path = st.text_input("MS-TCN checkpoint", value="runs/models/mstcn_tinyvirat/checkpoint.pt")
uploaded_video = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video and st.button("Analyze", type="primary"):
    procedure = Procedure.from_json(procedure_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as handle:
        handle.write(uploaded_video.getbuffer())
        video_path = handle.name

    progress = st.progress(0, text="Preparing video")
    if mode == "Temporal":
        architecture = AIOpsArchitectureConfig.from_json(architecture_path)
        checkpoint = checkpoint_path if checkpoint_path and Path(checkpoint_path).exists() else None
        progress.progress(25, text="Extracting clip features")
        payload = run_temporal_inference(video_path, procedure, architecture, checkpoint_path=checkpoint)
        progress.progress(75, text="Decoding temporal labels")
    else:
        payload = run_baseline_inference(video_path, procedure)
        progress.progress(75, text="Running baseline")
    progress.progress(100, text="Done")

    if payload.get("clip_predictions"):
        st.subheader("Clip Labels")
        st.dataframe(payload["clip_predictions"], use_container_width=True)
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
