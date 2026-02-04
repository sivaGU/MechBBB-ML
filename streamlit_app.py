"""
MechBBB Streamlit GUI. Two-stage mechanistically augmented BBB permeability classifier (Model C).

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.mechbbb
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DIR = PROJECT_ROOT

import io
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.mechbbb.predict import predict_single, predict_batch, load_predictor


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdbqt":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if "SMILES" in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "SMILES" in part.upper() and i + 1 < len(parts):
                            potential = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


def get_mol_with_3d(smiles: str, file_content: Optional[bytes] = None, file_extension: Optional[str] = None):
    """
    Get an RDKit mol with 3D coordinates for visualization.
    Uses uploaded file coords if present, else generates 3D from SMILES.
    Returns Chem.Mol or None.
    """
    mol = None
    if file_content is not None and file_extension is not None:
        ext = file_extension.lower()
        try:
            text = file_content.decode("utf-8")
            if ext == ".sdf":
                from io import StringIO
                supplier = Chem.SDMolSupplier(StringIO(text))
                mols = [m for m in supplier if m is not None]
                for m in mols:
                    if m.GetNumConformers() > 0:
                        mol = m
                        break
                else:
                    mol = mols[0] if mols else None
            elif ext == ".mol":
                mol = Chem.MolFromMolBlock(text)
            elif ext in (".pdb", ".pdbqt"):
                mol = Chem.MolFromPDBBlock(text)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2Block(text)
            if mol is not None and mol.GetNumConformers() == 0:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                mol = None
        except Exception:
            mol = None
    if mol is None and smiles:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumConformers() == 0:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception:
                return None
    return mol


def render_ligand_structure(mol, size: int = 400) -> Optional[bytes]:
    """
    Draw the ligand as a 2D chemical structure (atoms and bonds) using RDKit.
    Returns PNG image bytes or None on failure.
    """
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=(size, size))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MechBBB - BBB Permeability Studio",
    page_icon=None,
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/mechbbb-gui/issues",
        "About": "Two-stage mechanistically augmented BBB permeability classifier (Model C).",
    },
)

# Inject custom CSS for color palette - blue-teal scale
st.markdown("""
<style>
    /* Blue-teal palette: light powder -> midnight azure */
    :root {
        --light-powder-blue: #E0F4F8;
        --soft-sky-blue: #B3E5F0;
        --light-azure: #80D4E8;
        --medium-steel-blue: #4DB8D0;
        --deep-cerulean: #2A9DB5;
        --rich-teal-blue: #1E7A8C;
        --midnight-azure: #0D4F5C;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    section.main,
    .main,
    [data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    div[data-testid="stAppViewContainer"] > div > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    .main .block-container,
    section.main .block-container {
        background-color: #ffffff !important;
        padding: 2rem 3rem;
        margin: 2rem auto;
        max-width: 1400px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(13, 79, 92, 0.08);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D4F5C 0%, #0A3D47 100%);
        color: #ffffff;
        min-width: 200px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 280px !important;
        min-width: 200px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #0A3D47;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4DB8D0 0%, #2A9DB5 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(77, 184, 208, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        box-shadow: 0 4px 8px rgba(42, 157, 181, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
        box-shadow: 0 0 0 0.3rem rgba(77, 184, 208, 0.35);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%);
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%);
    }
    
    h1, h2, h3 {
        color: #1E7A8C;
        font-weight: 700;
    }
    
    a {
        color: #1E7A8C;
        text-decoration: none;
    }
    
    a:hover {
        color: #2A9DB5;
        text-decoration: underline;
    }
    
    [data-testid="stMetricValue"] {
        color: #1E7A8C;
        font-weight: 600;
    }
    
    .stSuccess {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stInfo {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        border-left: 4px solid #4DB8D0;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #2A9DB5;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stError {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
        border-left: 4px solid #1E7A8C;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stRadio > label,
    .stSelectbox > label,
    .stTextInput > label,
    .stSlider > label,
    .stFileUploader > label {
        color: #1E7A8C;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #E0F4F8 0%, #B3E5F0 100%);
        color: #1E7A8C;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #B3E5F0 0%, #80D4E8 100%);
    }
    
    .stDataFrame {
        border: 2px solid #4DB8D0;
        border-radius: 4px;
    }
    
    hr {
        border-color: #4DB8D0;
        border-width: 2px;
    }
    
    .stSlider .stSlider > div > div {
        background-color: #4DB8D0;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2A9DB5 0%, #1E7A8C 100%) !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #1E7A8C 0%, #0D4F5C 100%) !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.9);
    }
    
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.3) 0%, rgba(42, 157, 181, 0.25) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: linear-gradient(90deg, rgba(77, 184, 208, 0.25) 0%, rgba(42, 157, 181, 0.2) 100%);
        border-left: 4px solid #4DB8D0;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .main .block-container > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor():
    return load_predictor(HANDOFF_DIR)

DEFAULT_THRESHOLD = 0.35

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.title("MechBBB - Blood-Brain Barrier Permeability Studio")
    st.caption(
        "Two-stage mechanistically augmented BBB permeability classifier (Model C)."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** BBB permeability classification (Model C)
        - **Architecture:** Stage-1 (efflux/influx/PAMPA) + Stage-2 (PhysChem+ECFP+mech)
        - **Threshold:** 0.35 (MCC-optimal on BBBP validation)
        - **Status:** All pages available
        """
    )
    st.sidebar.success("Interactive ligand screening available!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams struggle to predict whether small molecules cross the blood-brain barrier.
        MechBBB (Model C) is a two-stage mechanistically augmented classifier that first predicts
        auxiliary ADME properties (efflux, influx, PAMPA) and then combines them with physicochemical
        and fingerprint features to predict BBB permeability. This approach improves both
        external generalization and interpretability.
        """
    )

    st.markdown(
        """
        ### Model highlights
        - **Stage-1:** LightGBM models trained on auxiliary mechanistic datasets (BBBP excluded) yield p_efflux, p_influx, p_pampa.
        - **Stage-2:** Model C = PhysChem + ECFP4 + mechanistic probs; ensemble of 5 seeds; threshold 0.35.
        - **No tuning on external data:** threshold selected on BBBP validation set only.
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to predict!** Use the **MechBBB Prediction** page in the sidebar to enter SMILES strings or upload a CSV file and get BBB permeability predictions with mechanistic probabilities."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home:** This overview
        - **Documentation:** Setup, model details, and usage
        - **MechBBB Prediction:** Run predictions (SMILES, structure files, or Batch CSV)
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the MechBBB Model C classifier.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for the MechBBB two-stage mechanistically augmented
        BBB permeability classifier (Model C). It supports single SMILES input, structure file upload (SDF, MOL, PDB, PDBQT, MOL2), and batch CSV processing.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application
        ├── requirements.txt      # Dependencies
        ├── src/mechbbb/          # Prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   └── cli.py            # Command-line interface
        └── artifacts/            # Model artifacts
            ├── stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib
            ├── stage2_modelC/    # model_seed0.pkl through model_seed4.pkl
            ├── threshold.json
            └── feature_config.json
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Launch the app: `streamlit run streamlit_app.py`.
        4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview (Model C)
        - **Stage-1:** LightGBM models on PhysChem + ECFP4 yield p_efflux, p_influx, p_pampa.
        - **Stage-2:** 5-model ensemble on PhysChem + ECFP4 + mechanistic probs yields P(BBB+).
        - **Threshold:** 0.35 (MCC-optimal on BBBP validation set).
        - **Features:** 10 physicochemical descriptors + 2048-bit ECFP4 + 3 mechanistic probabilities = 2061 total.
        """
    )

    st.markdown(
        """
        ## Uncertainty Quantification
        The model provides uncertainty estimates for each prediction:
        - **Standard Error:** Calculated from the variance across the 5-model ensemble (std_dev / √5).
        - **95% Confidence Interval:** P(BBB+) ± 2×SE, providing a range within which the true probability likely falls.
        - **Display:** Single predictions show the error range and confidence interval. Batch CSV outputs include columns for standard error, error percentage, and CI bounds.
        """
    )

    st.markdown(
        """
        ## CLI usage
        From the project folder:
        ```bash
        python -m src.mechbbb.cli --smiles "CCO" "c1ccccc1" --output out.csv
        python -m src.mechbbb.cli --input example_inputs.csv --output out.csv
        ```
        Output columns: smiles, canonical_smiles, prob_BBB+, prob_std_error, prob_std_error_pct, prob_CI_lower, prob_CI_upper, BBB_class, p_efflux, p_influx, p_pampa, threshold, error.
        """
    )

    st.success("Questions? Contact: Dr. Sivanesan Dakshanamurthy (sd233@georgetown.edu)")


def render_mechbbb_prediction_page():
    """Render the MechBBB prediction page."""
    st.title("BBB Permeability Prediction")
    st.markdown(
        """
        Predict BBB permeability using MechBBB (Model C). Enter a SMILES string, upload a structure file (SDF, MOL, PDB, PDBQT, MOL2), or upload a CSV file for batch processing.
        The model outputs P(BBB+), mechanistic probabilities (p_efflux, p_influx, p_pampa), and classification.
        
        **Input modes:** Single SMILES or structure file | Batch (CSV with smiles/SMILES column)
        """
    )

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info(
            "Ensure the **artifacts/** folder contains:\n"
            "- stage1_efflux.joblib, stage1_influx.joblib, stage1_pampa.joblib\n"
            "- stage2_modelC/ with model_seed0.pkl through model_seed4.pkl\n"
            "- threshold.json"
        )
        return

    st.sidebar.markdown("### Settings")
    threshold = st.sidebar.slider(
        "Classification threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01
    )
    st.sidebar.info(
        "**MechBBB (Model C)** Default threshold 0.35 = MCC-optimal on BBBP validation."
    )

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single SMILES or structure file", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single SMILES or structure file":
        smiles_input = st.text_input(
            "SMILES (or upload a structure file below)",
            placeholder="e.g. CCO, c1ccccc1",
            key="smiles_input",
        )
        st.markdown("**Or upload a structure file:**")
        structure_file = st.file_uploader(
            "Upload structure file",
            type=["sdf", "mol", "pdb", "pdbqt", "mol2"],
            key="structure_upload",
            help="Supported: SDF, MOL, PDB, PDBQT, MOL2. First molecule will be used.",
        )
        smiles_to_use = None
        if structure_file:
            content = structure_file.read()
            ext = os.path.splitext(structure_file.name)[1]
            extracted = extract_smiles_from_file(content, ext)
            if extracted:
                smiles_to_use = extracted
                # Store for 3D viewer (use uploaded structure if it has 3D coords)
                st.session_state.structure_file_content = content
                st.session_state.structure_file_ext = ext
                st.success(f"Extracted SMILES from {structure_file.name}")
            else:
                st.session_state.structure_file_content = None
                st.session_state.structure_file_ext = None
                st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
        elif smiles_input and smiles_input.strip():
            smiles_to_use = smiles_input.strip()
            st.session_state.structure_file_content = None
            st.session_state.structure_file_ext = None
        if st.button("Predict", type="primary", key="btn_single"):
            if smiles_to_use:
                result = predict_single(
                    smiles_to_use,
                    threshold=threshold,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("Valid SMILES")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Display P(BBB+) with error range
                        if result.prob_std_error is not None:
                            error_pct = result.prob_std_error * 100
                            # Use 2*SE for ~95% confidence interval
                            ci_lower = max(0.0, result.prob - 2 * result.prob_std_error)
                            ci_upper = min(1.0, result.prob + 2 * result.prob_std_error)
                            st.metric(
                                "P(BBB+)", 
                                f"{result.prob:.4f}",
                                help=f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
                            )
                            st.caption(f"± {error_pct:.2f}% (std error)")
                        else:
                            st.metric("P(BBB+)", f"{result.prob:.4f}")
                    with col2:
                        st.metric("Prediction", result.bbb_class)
                    with col3:
                        st.metric("Threshold", f"{threshold:.2f}")

                    # Display confidence interval details
                    if result.prob_std_error is not None:
                        st.markdown("#### Uncertainty Analysis")
                        err_col1, err_col2, err_col3 = st.columns(3)
                        with err_col1:
                            error_pct = result.prob_std_error * 100
                            st.metric("Standard Error", f"± {error_pct:.2f}%")
                        with err_col2:
                            ci_lower = max(0.0, result.prob - 2 * result.prob_std_error)
                            st.metric("95% CI Lower", f"{ci_lower:.4f}")
                        with err_col3:
                            ci_upper = min(1.0, result.prob + 2 * result.prob_std_error)
                            st.metric("95% CI Upper", f"{ci_upper:.4f}")
                        st.info(
                            f"**Prediction Range:** P(BBB+) = {result.prob:.4f} ± {result.prob_std_error:.4f} "
                            f"(95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}])"
                        )

                    st.subheader("Mechanistic probabilities")
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("p_efflux", f"{result.p_efflux:.4f}")
                    with mcol2:
                        st.metric("p_influx", f"{result.p_influx:.4f}")
                    with mcol3:
                        st.metric("p_pampa", f"{result.p_pampa:.4f}")

                    # Ligand structure (2D chemical drawing)
                    st.subheader("Ligand Structure")
                    file_content = st.session_state.get("structure_file_content")
                    file_ext = st.session_state.get("structure_file_ext")
                    mol = get_mol_with_3d(
                        result.canonical_smiles,
                        file_content=file_content,
                        file_extension=file_ext,
                    )
                    img_bytes = render_ligand_structure(mol) if mol else None
                    if img_bytes:
                        st.image(img_bytes, use_container_width=False, width=400)
                    else:
                        st.warning("Could not draw structure for this molecule.")
                else:
                    st.error(result.error)
            else:
                st.warning("Please enter a SMILES string or upload a structure file (SDF, MOL, PDB, PDBQT, MOL2).")

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
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
                st.error("CSV must have a SMILES column (smiles, SMILES, canonical_smiles, or smi).")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                if st.button("Predict batch", type="primary", key="btn_batch"):
                    smiles_list = df[col].astype(str).tolist()
                    results = predict_batch(
                        smiles_list,
                        threshold=threshold,
                        predictor=predictor,
                    )
                    df_out = df.copy()
                    df_out["prob_BBB+"] = [r.prob for r in results]
                    df_out["BBB_class"] = [r.bbb_class for r in results]
                    # Add standard error columns
                    df_out["prob_std_error"] = [
                        f"{r.prob_std_error:.6f}" if r.prob_std_error is not None else "" 
                        for r in results
                    ]
                    df_out["prob_std_error_pct"] = [
                        f"{r.prob_std_error * 100:.2f}%" if r.prob_std_error is not None else "" 
                        for r in results
                    ]
                    # Add confidence interval bounds
                    df_out["prob_CI_lower"] = [
                        f"{max(0.0, r.prob - 2 * r.prob_std_error):.6f}" 
                        if r.prob_std_error is not None else "" 
                        for r in results
                    ]
                    df_out["prob_CI_upper"] = [
                        f"{min(1.0, r.prob + 2 * r.prob_std_error):.6f}" 
                        if r.prob_std_error is not None else "" 
                        for r in results
                    ]
                    df_out["p_efflux"] = [r.p_efflux for r in results]
                    df_out["p_influx"] = [r.p_influx for r in results]
                    df_out["p_pampa"] = [r.p_pampa for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "mechbbb_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with a SMILES column to run batch predictions.")

    st.divider()
    st.caption(
        "MechBBB (Model C). Stage-1: efflux/influx/PAMPA. Stage-2: PhysChem+ECFP+mech."
    )


# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("MechBBB Prediction", use_container_width=True, key="nav_prediction"):
        st.session_state.current_page = "MechBBB Prediction"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "MechBBB Prediction":
        render_mechbbb_prediction_page()


if __name__ == "__main__":
    main()
