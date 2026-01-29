# CalcBB: BBB Permeability Prediction GUI

A production-ready Streamlit web application for predicting Blood-Brain Barrier (BBB) permeability using the validated **BBB HandOff** ensemble model artifacts. The interface allows users to upload ligand structure files and receive BBB permeability predictions with RDKit feature calculations.

Link: https://calcbb-gui-cpw77wkaat73mnjpn9hxdy.streamlit.app/

## Features

- **Single-ligand prediction interface** - Upload one ligand at a time for focused analysis
- **Multiple file format support** - Supports SDF, MOL, PDB, PDBQT, MOL2, and CSV formats
- **Automatic SMILES extraction** - Extracts SMILES strings from molecular structure files
- **RDKit feature computation** - Calculates 10 physicochemical descriptors + 2048-bit Morgan fingerprint (2058 total features)
- **BBB permeability prediction** - 5-model LightGBM ensemble with isotonic calibration (HandOff artifacts)
- **Applicability hint** - Reports max Tanimoto similarity to the training set (from HandOff `train_fps.npz`)
- **Clean, modern interface** - Red-themed aesthetic design with simplified displays
- **Model documentation** - Comprehensive documentation page with model details

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. Use the sidebar to navigate between pages.

## Model Artifacts (HandOff bundle)

The application requires the BBB HandOff artifact bundle to make predictions. **The app will still work for file parsing and descriptor computation without artifacts**, but predictions require the following files in the `artifacts/` directory:

```
artifacts/
├── feature_spec.json
├── model_config.json
├── operating_points.json
├── train_fps.npz
└── models/
    ├── ensemble_seed1.joblib
    ├── ensemble_seed2.joblib
    ├── ensemble_seed3.joblib
    ├── ensemble_seed4.joblib
    ├── ensemble_seed5.joblib
    └── isotonic.joblib
```

See `ARTIFACTS_GUIDE.md` for detailed instructions on obtaining or generating these artifacts.

## Project Structure

```
BBB FINAL GUI/
├── streamlit_app.py          # Main consolidated application (single file)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── ARTIFACTS_GUIDE.md        # Guide for obtaining model artifacts
├── DEPLOYMENT.md             # Deployment instructions
├── .gitignore                # Git ignore file
└── artifacts/                # Model artifacts directory (add your artifacts here)
    ├── feature_spec.json
    ├── model_config.json
    ├── operating_points.json
    ├── train_fps.npz
    └── models/
        ├── ensemble_seed1.joblib
        ├── ensemble_seed2.joblib
        ├── ensemble_seed3.joblib
        ├── ensemble_seed4.joblib
        ├── ensemble_seed5.joblib
        └── isotonic.joblib
```

## Usage

1. **Navigate to Ligand Prediction page** using the sidebar
2. **Upload a ligand file** in one of the supported formats:
   - SDF (Structure-Data File)
   - MOL (MDL Molfile)
   - PDB (Protein Data Bank format)
   - PDBQT (AutoDock format)
   - MOL2 (Tripos MOL2 format)
   - CSV (with 'smiles' column - only first ligand processed)
3. **View extracted SMILES** and molecular structure (if available)
4. **Compute descriptors** - Click the button to calculate RDKit descriptors
5. **View predictions** - If model artifacts are available, see BBB permeability predictions

## Model Details

The application uses the BBB HandOff artifact bundle:

- **Architecture**: 5 LightGBM models (ensemble averaged)
- **Calibration**: Isotonic regression (`models/isotonic.joblib`)
- **Features**: 10 RDKit physicochemical descriptors + 2048-bit Morgan fingerprint = 2058 total
- **Operating points**: Thresholds loaded from `operating_points.json`
- **Similarity**: Max Tanimoto similarity computed vs training fingerprints (`train_fps.npz`)

## Deployment

### Streamlit Cloud

1. Push this entire folder to a GitHub repository
2. Ensure `streamlit_app.py` and `requirements.txt` are in the root directory
3. Add model artifacts to the `artifacts/` directory in your repository
4. Connect repository to Streamlit Cloud
5. Deploy!

The application is a **single-file deployment** - all code is consolidated in `streamlit_app.py`.

See `DEPLOYMENT.md` for detailed deployment instructions.

## How It Works

1. **User uploads ligand file** → SMILES extracted from structure
2. **RDKit computes descriptors** → All available molecular descriptors calculated
3. **Descriptors displayed** → User can see all computed descriptor values
4. **Model makes predictions** → Two-stage ensemble predicts BBB permeability
5. **Results displayed** → Probabilities, mechanisms, and downloadable CSV

## Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`
- Model artifacts (optional - for predictions)

## Contact

Questions, bug reports, or collaboration requests: **Dr. Sivanesan Dakshanamurthy** - sd233@georgetown.edu

## License

See repository license file for details.




