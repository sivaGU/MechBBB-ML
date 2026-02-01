"""
MechBBB Streamlit GUI — Two-stage mechanistically augmented BBB permeability classifier (Model C).

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

# Ensure project root (this folder) is on path for src.mechbbb
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Artifact directory = this folder (contains artifacts/)
HANDOFF_DIR = PROJECT_ROOT

import streamlit as st
import pandas as pd

from src.mechbbb.predict import predict_single, predict_batch, load_predictor

st.set_page_config(page_title="MechBBB", page_icon="🧪", layout="wide")

st.title("🧪 MechBBB")
st.markdown("**Two-stage mechanistically augmented BBB permeability classifier** (Model C).")


@st.cache_resource
def get_predictor():
    return load_predictor(HANDOFF_DIR)


try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info(
        "Ensure the **artifacts/** folder contains:\n"
        "- stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib\n"
        "- stage2_modelC/ with model_seed0.pkl … model_seed4.pkl\n"
        "- threshold.json"
    )
    st.stop()

DEFAULT_THRESHOLD = 0.35

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Classification threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01
)
st.sidebar.info(
    "**MechBBB (Model C)** · Stage-1: efflux/influx/PAMPA · "
    "Stage-2: PhysChem+ECFP+mech · Default threshold 0.35"
)

st.header("Input")
input_mode = st.radio("Input mode", ["Single SMILES", "Batch (CSV)"], horizontal=True)

if input_mode == "Single SMILES":
    smiles_input = st.text_input("SMILES", placeholder="e.g. CCO, c1ccccc1")
    if st.button("Predict", type="primary") and smiles_input.strip():
        result = predict_single(
            smiles_input.strip(), threshold=threshold, predictor=predictor
        )
        if result.is_valid:
            st.success("✅ Valid SMILES")
            c1, c2, c3 = st.columns(3)
            c1.metric("P(BBB+)", f"{result.prob:.3f}")
            c2.metric("Prediction", result.bbb_class)
            c3.metric("Threshold", f"{threshold:.3f}")
            st.write(
                f"**p_efflux:** {result.p_efflux:.3f}  |  "
                f"**p_influx:** {result.p_influx:.3f}  |  "
                f"**p_pampa:** {result.p_pampa:.3f}"
            )
        else:
            st.error(f"❌ {result.error}")
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col = next(
            (
                c
                for c in df.columns
                if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"
            ),
            None,
        )
        if col is None:
            st.error("CSV must have a SMILES column")
        else:
            if st.button("Predict batch", type="primary"):
                smiles_list = df[col].astype(str).tolist()
                results = predict_batch(
                    smiles_list, threshold=threshold, predictor=predictor
                )
                df_out = df.copy()
                df_out["prob_BBB+"] = [r.prob for r in results]
                df_out["BBB_class"] = [r.bbb_class for r in results]
                df_out["p_efflux"] = [r.p_efflux for r in results]
                df_out["p_influx"] = [r.p_influx for r in results]
                df_out["p_pampa"] = [r.p_pampa for r in results]
                st.dataframe(df_out)
                st.download_button(
                    "Download CSV",
                    df_out.to_csv(index=False),
                    "mechbbb_predictions.csv",
                    "text/csv",
                )

st.caption(
    "MechBBB (Model C). Threshold 0.35 = MCC-optimal on BBBP validation. Artifacts in this folder."
)
