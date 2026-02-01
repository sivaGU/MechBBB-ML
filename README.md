# CalcBB - Two-Stage Mechanistically Augmented BBB Permeability Classifier (Model C)

A production-ready Streamlit GUI for predicting Blood-Brain Barrier (BBB) permeability using the validated **CalcBB Model C** two-stage ensemble.

## Features

- **Single SMILES prediction** — Type a SMILES string and get BBB permeability + mechanistic probabilities
- **Batch CSV prediction** — Upload a CSV with a SMILES column, download results
- **Two-stage Model C** — Stage-1: efflux/influx/PAMPA; Stage-2: PhysChem + ECFP4 + mechanistic probs
- **Adjustable threshold** — Default 0.35 (MCC-optimal on BBBP validation)

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If RDKit fails, try:
```bash
pip install rdkit
pip install lightgbm pandas numpy streamlit scikit-learn joblib
```

### 3. Run the Streamlit GUI

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
.
├── streamlit_app.py       # CalcBB GUI
├── requirements.txt       # Dependencies
├── src/
│   └── calcbb/            # Prediction module
│       ├── predict.py     # predict_single, predict_batch, load_predictor
│       └── cli.py         # Command-line interface
├── artifacts/             # Model artifacts
│   ├── stage1_efflux.joblib
│   ├── stage1_influx.joblib
│   ├── stage1_pampa.joblib
│   ├── stage2_modelC/     # model_seed0.pkl … model_seed4.pkl
│   ├── threshold.json
│   └── feature_config.json
├── example_inputs.csv
├── example_outputs.csv
└── PROCEDURES_FOR_FRIEND.md
```

## Usage

### GUI

- **Single SMILES:** Enter e.g. `CCO` or `c1ccccc1`, click **Predict**
- **Batch CSV:** Upload a CSV with a column named `smiles` or `SMILES`, click **Predict batch**, then **Download CSV**

### CLI

From the project root:

```bash
# Predict a few SMILES
python -m src.calcbb.cli --smiles "CCO" "c1ccccc1" --output out.csv

# Predict from CSV
python -m src.calcbb.cli --input example_inputs.csv --output out.csv
```

## Model Details

- **Stage-1:** LightGBM models on PhysChem + ECFP4 → p_efflux, p_influx, p_pampa
- **Stage-2:** 5-model ensemble on PhysChem + ECFP4 + mechanistic probs → P(BBB+)
- **Threshold:** 0.35 (MCC-optimal on BBBP validation)

## Requirements

- Python 3.9 or 3.10 (3.11/3.12 usually work)
- Dependencies in `requirements.txt`

## Contact

Dr. Sivanesan Dakshanamurthy — sd233@georgetown.edu
