"""
BBB Permeability Prediction Streamlit App
Consolidated single-file application for Streamlit Cloud deployment.

This app provides a GUI for predicting BBB permeability using the validated BBB HandOff ensemble model.
Users can upload ligand structure files and receive:
- RDKit-derived physicochemical descriptors (10) + Morgan fingerprint features (2048 bits)
- Calibrated BBB permeability probability from an ensemble of 5 LightGBM models
"""

import json
import math
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, inchi, rdMolDescriptors

# Try to import Draw module - may fail on systems without X11 libraries (e.g., Streamlit Cloud)
try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    # Draw module not available - visualization will be disabled
    Draw = None
    DRAW_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="BBB Permeability Studio",
    page_icon=None,
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/bbb-gui/issues",
        "About": "Sparse-label multi-task learning workflow for BBB permeability modelling.",
    },
)

# Inject custom CSS for color palette
st.markdown("""
<style>
    /* Color Palette: Light red theme */
    :root {
        --primary-color: #FF6B6B;
        --primary-light: #FF8E8E;
        --primary-lighter: #FFB3B3;
        --accent-1: #FF9999;
        --accent-2: #FFB3B3;
        --accent-3: #FFCCCC;
        --accent-4: #FFE0E0;
        --accent-5: #FFE5E5;
        --accent-6: #FFF0F0;
        --very-light-red: #FFF5F5;
    }
    
    /* Page background - very light red */
    .stApp {
        background-color: #FFF5F5;
    }
    
    /* Main background with margins */
    .main .block-container {
        background-color: #ffffff;
        padding: 2rem 3rem;
        margin: 2rem auto;
        max-width: 1400px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.1);
    }
    
    /* Sidebar - light red */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFE5E5 0%, #FFD5D5 100%);
        color: #333333;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #FFE0E0;
    }
    
    /* Primary buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(255, 107, 107, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF8E8E 0%, #FF9999 100%);
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        background: linear-gradient(135deg, #FF9999 0%, #FFB3B3 100%);
        box-shadow: 0 0 0 0.3rem rgba(255, 107, 107, 0.3);
    }
    
    /* Secondary buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #FF8E8E 0%, #FF9999 100%);
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #FF9999 0%, #FFB3B3 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FF6B6B;
        font-weight: 700;
    }
    
    /* Links */
    a {
        color: #FF6B6B;
        text-decoration: none;
    }
    
    a:hover {
        color: #FF8E8E;
        text-decoration: underline;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #FF6B6B;
        font-weight: 600;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(90deg, #FFE5E5 0%, #FFD5D5 100%);
        border-left: 4px solid #FF6B6B;
        color: #333333;
        border-radius: 4px;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(90deg, #FFE5E5 0%, #FFD5D5 100%);
        border-left: 4px solid #FF6B6B;
        color: #333333;
        border-radius: 4px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(90deg, #FFE0E0 0%, #FFD5D5 100%);
        border-left: 4px solid #FF8E8E;
        color: #333333;
        border-radius: 4px;
    }
    
    /* Error boxes */
    .stError {
        background: linear-gradient(90deg, #FFD5D5 0%, #FFCCCC 100%);
        border-left: 4px solid #FF6B6B;
        color: #333333;
        border-radius: 4px;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* Selectbox */
    .stSelectbox > label {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* Text input labels */
    .stTextInput > label {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* Slider */
    .stSlider > label {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* File uploader */
    .stFileUploader > label {
        color: #FF6B6B;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #FFE5E5 0%, #FFD5D5 100%);
        color: #333333;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #FFD5D5 0%, #FFCCCC 100%);
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 2px solid #FF6B6B;
        border-radius: 4px;
    }
    
    /* Dividers */
    hr {
        border-color: #FF6B6B;
        border-width: 2px;
    }
    
    /* Slider track */
    .stSlider .stSlider > div > div {
        background-color: #FF6B6B;
    }
    
    /* Navigation buttons - cleaner styling */
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h3 {
        color: #FF6B6B;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: #333333;
    }
    
    /* Sidebar success/info boxes */
    [data-testid="stSidebar"] .stSuccess {
        background: linear-gradient(90deg, #FFE5E5 0%, #FFD5D5 100%);
        border-left: 4px solid #FF6B6B;
        color: #333333;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: linear-gradient(90deg, #FFE5E5 0%, #FFD5D5 100%);
        border-left: 4px solid #FF6B6B;
        color: #333333;
    }
    
    /* Sidebar spacing */
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(255, 107, 107, 0.3);
    }
    
    /* Ensure main content has white background for readability */
    .main .block-container > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# BBB MODEL - FEATURES + ARTIFACTS (HandOff format)
# ============================================================================

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _HAS_STANDARDIZE = True
except Exception:
    _HAS_STANDARDIZE = False


@dataclass
class MolId:
    canonical_smiles: str
    inchikey: str


def _cleanup_mol(mol: Chem.Mol) -> Chem.Mol:
    if not _HAS_STANDARDIZE:
        return mol
    mol = rdMolStandardize.Cleanup(mol)
    parent = rdMolStandardize.FragmentParent(mol)
    uncharger = rdMolStandardize.Uncharger()
    parent = uncharger.uncharge(parent)
    return parent


def standardize_and_identify(smiles: Optional[str], inchi_str: Optional[str] = None) -> Optional[MolId]:
    mol = None
    if smiles and isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles)
    if mol is None and inchi_str and isinstance(inchi_str, str) and inchi_str.startswith("InChI="):
        try:
            mol = inchi.MolFromInchi(inchi_str, sanitize=True, removeHs=True)
        except Exception:
            mol = None
    if mol is None:
        return None
    try:
        mol = _cleanup_mol(mol)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        ik = inchi.MolToInchiKey(mol)
        if not ik:
            return None
        return MolId(canonical_smiles=can, inchikey=ik)
    except Exception:
        return None


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2
    """
    try:
        if file_extension.lower() == '.sdf':
            # SDF file
            from io import StringIO
            sdf_data = StringIO(file_content.decode('utf-8'))
            supplier = Chem.SDMolSupplier(sdf_data)
            mol = None
            for m in supplier:
                if m is not None:
                    mol = m
                    break
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        
        elif file_extension.lower() == '.mol':
            # MOL file
            mol = Chem.MolFromMolBlock(file_content.decode('utf-8'))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        
        elif file_extension.lower() == '.pdb':
            # PDB file - try to read with RDKit
            mol = Chem.MolFromPDBBlock(file_content.decode('utf-8'))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            # If RDKit fails, try to extract SMILES from REMARK lines if present
            lines = file_content.decode('utf-8').split('\n')
            for line in lines:
                if 'SMILES' in line.upper() or 'REMARK' in line:
                    # Try to find SMILES in REMARK
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'SMILES' in part.upper() and i + 1 < len(parts):
                            potential_smiles = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential_smiles)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        
        elif file_extension.lower() == '.pdbqt':
            # PDBQT file (AutoDock format) - similar to PDB
            # Try to read as PDB first
            mol = Chem.MolFromPDBBlock(file_content.decode('utf-8'))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            # PDBQT may have SMILES in comments
            lines = file_content.decode('utf-8').split('\n')
            for line in lines:
                if 'SMILES' in line.upper():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'SMILES' in part.upper() and i + 1 < len(parts):
                            potential_smiles = parts[i + 1]
                            mol = Chem.MolFromSmiles(potential_smiles)
                            if mol:
                                return Chem.MolToSmiles(mol, canonical=True)
        
        elif file_extension.lower() == '.mol2':
            # MOL2 file - RDKit doesn't support directly, try basic parsing
            # This is a simplified parser - may not work for all MOL2 files
            content = file_content.decode('utf-8')
            # Try to find SMILES in comments or use atom/bond info
            # For now, return None and let user know
            return None
        
        elif file_extension.lower() == '.csv':
            # CSV file - should have been handled separately
            return None
            
    except Exception as e:
        return None
    
    return None


def compute_physchem10(smiles: str) -> pd.Series:
    """Compute the 10 HandOff physicochemical RDKit descriptors for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES for physchem computation")
    return pd.Series(
        {
            "MolWt": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
            "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    )


def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute 2048-bit Morgan fingerprint (binary) for one molecule."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def compute_features_2058(canon_smiles: str, feature_spec: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute HandOff features:
    - 10 physchem descriptors (DataFrame 1 row)
    - 2048 Morgan bits (numpy 1D)
    Returns (physchem_df, features_1d)
    """
    physchem_cols: List[str] = feature_spec["physchem_cols"]
    morgan_cfg = feature_spec["morgan"]
    n_bits = int(morgan_cfg["n_bits"])
    radius = int(morgan_cfg["radius"])

    phys = compute_physchem10(canon_smiles)[physchem_cols]
    fp = compute_morgan_fp(canon_smiles, radius=radius, n_bits=n_bits)
    features = np.hstack([phys.values.astype(float), fp.astype(np.uint8)])
    return phys.to_frame().T, features


def compute_similarity_max(query_fp: np.ndarray, train_fps: np.ndarray) -> float:
    """Compute maximum Tanimoto similarity of query_fp to train_fps."""
    q = query_fp.astype(bool)
    tf = train_fps.astype(bool)
    if q.sum() == 0:
        return 0.0
    intersection = np.logical_and(q[None, :], tf).sum(axis=1)
    union = np.logical_or(q[None, :], tf).sum(axis=1)
    sims = np.where(union > 0, intersection / union, 0.0)
    return float(sims.max())

# ============================================================================
# ARTIFACT LOADING + PREDICTION (HandOff ensemble + isotonic)
# ============================================================================

@st.cache_resource
def load_handoff_artifacts(artifacts_dir: str):
    """Load HandOff artifact bundle (cached)."""
    base = os.path.abspath(artifacts_dir)
    feature_spec = json.load(open(os.path.join(base, "feature_spec.json"), "r", encoding="utf-8"))
    model_config = json.load(open(os.path.join(base, "model_config.json"), "r", encoding="utf-8"))
    operating_points = json.load(open(os.path.join(base, "operating_points.json"), "r", encoding="utf-8"))

    models = []
    for seed in range(1, 6):
        models.append(joblib.load(os.path.join(base, "models", f"ensemble_seed{seed}.joblib")))
    calibrator = joblib.load(os.path.join(base, "models", "isotonic.joblib"))
    train_fps = np.load(os.path.join(base, "train_fps.npz"))["fp"]
    return feature_spec, model_config, operating_points, models, calibrator, train_fps


def predict_bbb_handoff(
    canon_smiles: str,
    feature_spec: Dict[str, Any],
    operating_points: Dict[str, Any],
    models: List[Any],
    calibrator: Any,
    train_fps: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - result_df: single-row DataFrame with prediction fields
    - physchem_df: single-row DataFrame with 10 descriptor values
    """
    physchem_df, feats = compute_features_2058(canon_smiles, feature_spec)
    total_dim = int(feature_spec.get("total_dim", 2058))
    if feats.shape[0] != total_dim:
        raise ValueError(f"Feature dimension mismatch: {feats.shape[0]} != {total_dim}")

    X = feats.reshape(1, -1)
    probs_raw = [float(m.predict(X)[0]) for m in models]
    prob_raw_mean = float(np.mean(probs_raw))
    prob_cal = float(calibrator.transform([prob_raw_mean])[0])

    # similarity uses Morgan bits only
    n_phys = len(feature_spec["physchem_cols"])
    query_fp = feats[n_phys:].astype(np.uint8)
    sim_max = compute_similarity_max(query_fp, train_fps)
    sim_flag = "low" if sim_max < 0.3 else "ok"

    thr_mcc = operating_points.get("mcc_max", None)
    thr_sens = operating_points.get("sens_90_spec_max", None)
    decision_mccmax = int(prob_cal >= thr_mcc) if thr_mcc is not None else None
    decision_highsens = int(prob_cal >= thr_sens) if thr_sens is not None else None

    out = pd.DataFrame(
        [
            {
                "canonical_smiles": canon_smiles,
                "prob_raw": prob_raw_mean,
                "prob_cal": prob_cal,
                "similarity_max": sim_max,
                "similarity_flag": sim_flag,
                "decision_mccmax": decision_mccmax,
                "decision_highsens": decision_highsens,
            }
        ]
    )
    return out, physchem_df

# ============================================================================
# STREAMLIT PAGES
# ============================================================================

def load_predictor(artifacts_dir: str = "artifacts"):
    """Load HandOff predictor artifacts (cached)."""
    try:
        if not os.path.exists(artifacts_dir):
            return None, f"Artifacts directory '{artifacts_dir}' not found."
        bundle = load_handoff_artifacts(artifacts_dir)
        return bundle, None
    except Exception as e:
        return None, f"Error loading artifacts: {str(e)}\n\n{traceback.format_exc()}"


def render_home_page():
    """Render the home/dashboard page."""
    st.title("Blood-Brain Barrier (BBB) Permeability Studio")
    st.caption(
        "Prototype interface for the sparse-label multi-task ensemble described in the BBB manuscript."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Model focus:** Calibrated BBB permeability classification  
        - **Architecture:** Masked multi-task ensemble blended with a single-task baseline  
        - **External validation:** BBBP and out-of-source (OOS) panels  
        - **Status:** All pages available
        """
    )
    st.sidebar.success("Interactive ligand screening available!")

    st.markdown(
        """
        ## Why this app exists
        Drug discovery teams struggle to predict whether small molecules cross the blood-brain barrier.
        The manuscript's fourth tab introduces a sparse-label multi-task (MT) learning workflow that blends
        auxiliary ADME tasks (PAMPA, PPB, efflux) with a calibrated single-task (ST) baseline. The blended
        predictor improves both external generalization and probability calibration, addressing two recurring
        issues in BBB screening campaigns.
        """
    )

    st.markdown(
        """
        ### Model highlights from Tab 4
        - **Sparse-label MT training:** Each auxiliary task contributes signal only where assays exist, avoiding label deletion or imputation bias.
        - **Stacked calibration:** MT logits are linearly blended with the ST baseline before post-hoc calibration selected on the development fold.
        - **Reproducibility guardrails:** All tables/figures originate from `results/metrics_clean_fixed.json`, with scripted pipelines and stratified bootstraps (B = 2000).
        """
    )

    st.divider()

    st.markdown("## Performance at a glance (Tab 4 metrics)")

    internal_metrics = {"PR-AUC": 0.915, "ROC-AUC": 0.864, "ΔPR-AUC vs ST": "+0.102"}
    external_metrics = [
        {
            "dataset": "BBBP",
            "PR-AUC": 0.950,
            "ΔPR-AUC vs ST": "+0.155",
            "p-value": "< 0.001",
        },
        {
            "dataset": "Out-of-source (OOS)",
            "PR-AUC": 0.944,
            "ΔPR-AUC vs ST": "+0.185",
            "p-value": "< 0.001",
        },
    ]

    col_internal, col_ext_1, col_ext_2 = st.columns(3)
    with col_internal:
        st.metric("Internal PR-AUC", internal_metrics["PR-AUC"], internal_metrics["ΔPR-AUC vs ST"])
        st.metric("Internal ROC-AUC", internal_metrics["ROC-AUC"])

    with col_ext_1:
        st.metric("BBBP PR-AUC", external_metrics[0]["PR-AUC"], external_metrics[0]["ΔPR-AUC vs ST"])
        st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[0]['p-value']}")

    with col_ext_2:
        st.metric(
            "OOS PR-AUC",
            external_metrics[1]["PR-AUC"],
            external_metrics[1]["ΔPR-AUC vs ST"],
        )
        st.caption(f"One-sided ΔPR-AUC p-value {external_metrics[1]['p-value']}")

    st.markdown(
        """
        Calibration improves alongside discrimination: the blended model reports lower Brier score and
        expected calibration error (ECE) than the single-task baseline, with reliability diagrams approaching
        the identity line across internal and external datasets.
        """
    )

    st.divider()

    st.markdown("## From Tab 5: evaluation protocol & upcoming assets")

    tab5_col1, tab5_col2 = st.columns(2)
    with tab5_col1:
        st.subheader("Evaluation blueprint")
        st.markdown(
            """
            - **Primary metric:** Precision-recall AUC (PR-AUC); ROC-AUC reported as a secondary view.  
            - **Uncertainty:** Stratified bootstrap (B = 2000, seed = 42) yields 95% confidence intervals and ΔPR-AUC hypothesis tests.  
            - **Calibration checks:** Brier score, ECE, and reliability diagrams with equal-mass binning; Platt vs isotonic selected on the development fold.  
            - **Applicability domain:** Coverage vs precision curves using ensemble variance or representation distance thresholds.  
            """
        )

    with tab5_col2:
        st.subheader("Assets in progress")
        st.markdown(
            """
            - External & internal ROC/PR curves with confidence bands  
            - Calibration dashboards (reliability diagrams, ΔECE summaries)  
            - Confusion matrices at 0.5 and Youden thresholds  
            - Feature attribution (SHAP) views for top ADME descriptors  
            - Applicability domain plots showing precision vs coverage trade-offs  
            """
        )

    st.info(
        "**Ready to predict!** Use the 'Ligand Prediction' page in the sidebar to upload your SMILES strings or CSV files and get BBB permeability predictions."
    )

    st.markdown(
        """
        ---
        ### Roadmap
        1. **Completed** – Communication spine: home and documentation pages summarizing the Tab 4-5 manuscript content.  
        2. **Completed** – Ligand intake tab with SMILES/CSV upload, descriptor generation, and model scoring (see Ligand Prediction page).  
        3. **Planned** – Calibration overlay for user-submitted batches and automated report exports (PDF/CSV).
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material derived from Tabs 4-5 of the BBB manuscript.")

    st.markdown(
        """
        ## Purpose
        This application packages the manuscript's sparse-label multi-task (MT) modelling workflow into a
        Streamlit interface. The current release focuses on communication: summarising study context,
        evaluation methodology, and planned visual assets before the ligand submission module comes online.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py         # Single consolidated app file
        ├── requirements.txt         # Dependencies
        └── artifacts/               # Model artifacts (not in repo)
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
        ## Model overview (Tab 4 recap)
        - **Training data:** BBB permeability labels plus auxiliary ADME assays (PAMPA, PPB, efflux).  
        - **Learning strategy:** Masked MT ensemble blended with an ST baseline; losses applied only where task labels exist.  
        - **Calibration:** Platt vs isotonic assessed on the development fold; chosen calibrator reused for internal/external evaluations.  
        - **Reproducibility:** All metrics/figures regenerate from `results/metrics_clean_fixed.json`; stratified bootstrap (B = 2000) underpins confidence intervals and ΔPR-AUC tests.
        """
    )

    st.markdown(
        """
        ## Evaluation protocol (Tab 5 recap)
        - **Primary metric:** PR-AUC (robust to class imbalance); ROC-AUC reported for context.  
        - **Operational thresholds:** Summaries at 0.5 and Youden (≈0.793) include accuracy, sensitivity, specificity, F1, and MCC.  
        - **Calibration diagnostics:** Brier score, expected calibration error (ECE), reliability diagrams with equal-mass bins.  
        - **Applicability domain:** Precision vs coverage curves thresholded on ensemble variance or representation distance.  
        - **Feature interpretation:** SHAP beeswarm/waterfall plots planned for top descriptors (LightGBM head).
        """
    )

    st.markdown(
        """
        ## Roadmap
        - **Completed:** Ligand intake tab supporting SMILES/CSV uploads, descriptor generation, and scoring.  
        - **Planned visual assets:** External/internal ROC & PR with CI bands, calibration dashboards, confusion matrices, SHAP explorer, AD curves.  
        - **Reporting:** Automated PDF/CSV exports once ligand scoring is active.
        """
    )

    st.info(
        "Need to contribute? Fork the GitHub repository, branch from `main`, and submit a pull request. "
        "Include before/after screenshots when adding new widgets to keep the review focused."
    )

    st.success("Questions? Open an issue via the menu or tag the modelling team on Slack.")


def render_ligand_prediction_page():
    """Render the ligand prediction page."""
    st.title("BBB Permeability Prediction")
    st.markdown(
        """
        Upload a single ligand file to get BBB permeability predictions.
        The model uses RDKit to calculate physicochemical descriptors + Morgan fingerprints and predicts BBB permeability
        using an ensemble of 5 LightGBM models with isotonic calibration (HandOff artifact bundle).
        
        **Supported file formats:** SDF, MOL, PDB, PDBQT, MOL2, CSV (first row only)
        """
    )

    # Check if artifacts directory exists
    artifacts_dir = "artifacts"
    predictor_bundle = None
    error_msg = None
    
    if not os.path.exists(artifacts_dir):
        st.info(
            f"**Model artifacts not found.**\n\n"
            f"The '{artifacts_dir}' directory is missing. **You can still use this app to:**\n"
            f"- Extract and standardize SMILES\n"
            f"- Compute RDKit descriptors (physchem10 + Morgan fingerprint)\n"
            f"- Validate and canonicalize SMILES strings\n"
            f"- View molecular structures (if available)\n\n"
            f"**To enable BBB predictions, you need to add the model artifacts to your repository:**\n"
            f"- Create an `{artifacts_dir}/` directory in your repo\n"
            f"- Add the following files:\n"
            f"  - `{artifacts_dir}/feature_spec.json`\n"
            f"  - `{artifacts_dir}/model_config.json`\n"
            f"  - `{artifacts_dir}/operating_points.json`\n"
            f"  - `{artifacts_dir}/train_fps.npz`\n"
            f"  - `{artifacts_dir}/models/ensemble_seed1.joblib`\n"
            f"  - `{artifacts_dir}/models/ensemble_seed2.joblib`\n"
            f"  - `{artifacts_dir}/models/ensemble_seed3.joblib`\n"
            f"  - `{artifacts_dir}/models/ensemble_seed4.joblib`\n"
            f"  - `{artifacts_dir}/models/ensemble_seed5.joblib`\n"
            f"  - `{artifacts_dir}/models/isotonic.joblib`\n\n"
            f"**For Streamlit Cloud:** Upload these files to your GitHub repository in the `{artifacts_dir}/` folder."
        )
        artifacts_dir = st.text_input("Or specify a different artifacts directory path:", value="artifacts", key="artifacts_dir")
        
        # Try to load if user specified a different path
        if artifacts_dir and os.path.exists(artifacts_dir):
            predictor_bundle, error_msg = load_predictor(artifacts_dir)
    else:
        st.success(f"Found artifacts directory: `{artifacts_dir}`")
        predictor_bundle, error_msg = load_predictor(artifacts_dir)

    if error_msg:
        st.warning(f"Could not load model: {error_msg}")
        st.info("**Note:** You can still extract/standardize SMILES and compute descriptors without the model.")
        predictor_bundle = None
    elif predictor_bundle is not None:
        st.success(f"Model loaded successfully! BBB predictions are now available.")

    st.divider()

    # File upload - single ligand only
    uploaded_file = st.file_uploader(
        "Upload ligand file:",
        type=["sdf", "mol", "pdb", "pdbqt", "mol2", "csv"],
        help="Upload a single ligand file. Supported formats: SDF, MOL, PDB, PDBQT, MOL2, or CSV (with 'smiles' column).",
        key="ligand_upload"
    )

    smiles = None
    file_name = None

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1]
        
        try:
            file_content = uploaded_file.read()
            
            # Extract SMILES based on file type
            if file_extension.lower() == '.csv':
                # CSV file - extract SMILES from column
                df_upload = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer
                
                # Find SMILES column (case-insensitive)
                smiles_col = None
                for col in df_upload.columns:
                    if col.lower() == "smiles":
                        smiles_col = col
                        break
                
                if smiles_col is None:
                    st.error(f"No 'smiles' column found in CSV. Available columns: {', '.join(df_upload.columns)}")
                else:
                    smiles_list = df_upload[smiles_col].astype(str).tolist()
                    if len(smiles_list) > 1:
                        st.warning(f"CSV contains {len(smiles_list)} ligands. Only the first ligand will be processed.")
                    if len(smiles_list) > 0:
                        smiles = smiles_list[0]
            else:
                # Molecular structure file
                with st.spinner(f"Extracting SMILES from {file_extension.upper()} file..."):
                    extracted_smiles = extract_smiles_from_file(file_content, file_extension)
                    
                    if extracted_smiles:
                        smiles = extracted_smiles
                        st.success(f"Successfully extracted SMILES from {file_name}")
                    else:
                        st.error(f"Could not extract SMILES from {file_extension.upper()} file. The file may be corrupted or in an unsupported format.")
                        st.info("**Tip:** For PDB/PDBQT files, ensure the file contains valid molecular structure data. For MOL2 files, conversion may not be supported.")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # Process and validate SMILES
    if smiles:
        st.divider()
        
        with st.spinner("Standardizing and identifying molecule..."):
            mol_id = standardize_and_identify(smiles)

        if mol_id is None:
            st.error("Could not parse/standardize the molecule. Please check your file and try again.")
        else:
            canon_smiles = mol_id.canonical_smiles
            st.success("Ligand parsed successfully and ready for prediction")
            
            # Show molecule preview (only if Draw module is available)
            if DRAW_AVAILABLE:
                st.subheader("Molecular Structure")
                try:
                    mol = Chem.MolFromSmiles(canon_smiles)
                    if mol and Draw is not None:
                        img = Draw.MolToImage(mol, size=(400, 400))
                        st.image(img, caption=f"SMILES: {canon_smiles}", use_container_width=False)
                except Exception as e:
                    st.warning(f"Could not render molecular structure: {e}")
            elif not DRAW_AVAILABLE:
                st.info("Molecular structure visualization is not available on this platform.")
            
            # Show identifiers
            st.subheader("Molecule Identifiers")
            st.markdown(f"**Canonical SMILES:** `{canon_smiles}`")
            st.markdown(f"**InChIKey:** `{mol_id.inchikey}`")
            
            # Compute descriptors
            button_text = "Compute Descriptors & Make Predictions" if predictor_bundle else "Compute Descriptors"
            if st.button(button_text, type="primary"):
                with st.spinner("Computing RDKit descriptors and making predictions..."):
                    try:
                        # Always compute HandOff physchem10 + Morgan (2058 features)
                        if predictor_bundle is not None:
                            feature_spec, model_config, operating_points, models, calibrator, train_fps = predictor_bundle
                        else:
                            # still allow descriptor computation without model by using feature spec defaults
                            feature_spec = {
                                "physchem_cols": [
                                    "MolWt",
                                    "TPSA",
                                    "MolLogP",
                                    "NumHDonors",
                                    "NumHAcceptors",
                                    "NumRotatableBonds",
                                    "RingCount",
                                    "HeavyAtomCount",
                                    "FractionCSP3",
                                    "NumAromaticRings",
                                ],
                                "morgan": {"radius": 2, "n_bits": 2048, "use_counts": False},
                                "total_dim": 2058,
                            }

                        physchem_df, feats = compute_features_2058(canon_smiles, feature_spec)

                        st.subheader("Descriptor Information")
                        st.info("Computed 10 physicochemical descriptors and a 2048-bit Morgan fingerprint (2058 total features).")

                        st.subheader("Physicochemical Descriptor Values")
                        phys_t = physchem_df.iloc[0].to_frame(name="Value")
                        phys_t.index.name = "Descriptor"
                        st.dataframe(phys_t, use_container_width=True)

                        # Make predictions if model is available
                        if predictor_bundle is None:
                            st.info(
                                "**Descriptors computed successfully!**\n\n"
                                "To enable BBB permeability predictions, please add the model artifacts to your repository. "
                                "See the instructions at the top of this page for details."
                            )
                            # Option to download physchem descriptors
                            desc_csv = physchem_df.to_csv(index=False)
                            st.download_button(
                                label="Download Descriptors as CSV",
                                data=desc_csv,
                                file_name="bbb_physchem_descriptors.csv",
                                mime="text/csv",
                                key="download_descriptors"
                            )
                        else:
                            feature_spec, model_config, operating_points, models, calibrator, train_fps = predictor_bundle
                            result_df, _physchem_df = predict_bbb_handoff(
                                canon_smiles=canon_smiles,
                                feature_spec=feature_spec,
                                operating_points=operating_points,
                                models=models,
                                calibrator=calibrator,
                                train_fps=train_fps,
                            )

                            r = result_df.iloc[0]
                            prob_cal = float(r["prob_cal"])
                            prob_raw = float(r["prob_raw"])
                            sim_max = float(r["similarity_max"])
                            sim_flag = str(r["similarity_flag"])

                            thr_mcc = operating_points.get("mcc_max", None)
                            thr_sens = operating_points.get("sens_90_spec_max", None)

                            st.subheader("BBB Permeability Predictions")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Calibrated Probability", f"{prob_cal:.4f}")
                            with col2:
                                st.metric("Raw Probability (Ensemble Mean)", f"{prob_raw:.4f}")
                            with col3:
                                st.metric("Max Similarity to Training", f"{sim_max:.4f}", sim_flag)

                            st.subheader("Decisions")
                            dcol1, dcol2 = st.columns(2)
                            with dcol1:
                                if thr_mcc is not None and r["decision_mccmax"] is not None:
                                    st.metric(
                                        f"MCC Max Threshold (t={float(thr_mcc):.4f})",
                                        "BBB+" if int(r["decision_mccmax"]) == 1 else "BBB-",
                                    )
                                else:
                                    st.info("MCC threshold not available in operating_points.json")
                            with dcol2:
                                if thr_sens is not None and r["decision_highsens"] is not None:
                                    st.metric(
                                        f"High Sensitivity Threshold (t={float(thr_sens):.4f})",
                                        "BBB+" if int(r["decision_highsens"]) == 1 else "BBB-",
                                    )
                                else:
                                    st.info("High-sensitivity threshold not available in operating_points.json")

                            if sim_flag == "low":
                                st.warning(
                                    "Low similarity to the training set (similarity < 0.3). Predictions may be less reliable."
                                )

                            st.subheader("Download Results")
                            download_df = result_df.copy()
                            download_df["inchikey"] = mol_id.inchikey
                            download_df["uploaded_filename"] = file_name
                            csv = download_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prediction as CSV",
                                data=csv,
                                file_name="bbb_prediction.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                                
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        with st.expander("Error details", expanded=False):
                            st.code(traceback.format_exc())
    else:
        st.info("Please upload a ligand file to get started.")

    # Footer
    st.divider()
    st.markdown(
        """
        **Note:** This interface uses RDKit to compute features directly from molecular structure.
        The HandOff model uses 10 physicochemical descriptors and a 2048-bit Morgan fingerprint (2058 total features).
        """
    )


# ============================================================================
# MAIN APP - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    # Initialize session state for page navigation
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation with buttons
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")
    
    # Navigation buttons - use container width for full-width buttons
    if st.sidebar.button("🏠 Home", use_container_width=True, 
                         key="nav_home"):
        st.session_state.current_page = "Home"
    
    if st.sidebar.button("📚 Documentation", use_container_width=True,
                         key="nav_docs"):
        st.session_state.current_page = "Documentation"
    
    if st.sidebar.button("🧪 Ligand Prediction", use_container_width=True,
                         key="nav_prediction"):
        st.session_state.current_page = "Ligand Prediction"
    
    st.sidebar.markdown("---")
    
    # Render selected page
    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "Ligand Prediction":
        render_ligand_prediction_page()


if __name__ == "__main__":
    main()
