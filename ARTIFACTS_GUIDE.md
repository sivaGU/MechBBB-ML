# How to Get Model Artifacts

This GUI is wired to the **BBB HandOff** artifact bundle (ensemble models + isotonic calibrator + specs).
To enable predictions, you need to place that bundle into `BBB FINAL GUI/artifacts/`.

## Option 1: Use the provided HandOff artifacts (recommended)

You already have a verified artifact bundle here:
`C:\Users\madas\Downloads\BBB-Prediction-main\BBB HandOff\artifacts`

Copy the entire folder contents into `BBB FINAL GUI/artifacts/` so it looks like:

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

## Option 2: Train the Models Yourself

If you have access to the original training pipeline and data, you can regenerate the HandOff artifacts.
The GUI expects the same artifact schema and feature dimensions (2058).

### Requirements:

1. **Training data CSV files** with two columns:
   - `smiles`: SMILES strings of molecules
   - `label`: Binary labels (0 or 1)
   
   You need the following datasets:
   - `bbb_internal.csv` - BBB internal dataset
   - `efflux.csv` - Efflux mechanism dataset
   - `influx.csv` - Influx mechanism dataset
   - `pampa.csv` - PAMPA dataset
   - `cns.csv` - CNS dataset
   - `bbbp_external.csv` - (optional) External validation dataset

2. **Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install hyperopt  # Required for hyperparameter optimization
   ```

### Steps to Train:

1. **Prepare your training data:**
   - Create a `data/` directory (or specify your own)
   - Place all CSV files with `smiles` and `label` columns in that directory

2. **Run the training script:**
   - You'll need access to the training code from the original Colab notebook
   - Or use the training code referenced in the manuscript

3. **Wait for training to complete:**
   - Hyperparameter optimization can take several hours
   - Model training will follow
   - Artifacts will be saved to `artifacts/` directory

### Training Parameters:

- `n_trials`: Number of hyperparameter optimization trials (default: 100, can take hours)
- `n_splits`: Cross-validation folds (default: 5)
- `seed`: Random seed for reproducibility (default: 42)

## Verify Artifacts

After obtaining artifacts, verify they exist:

```bash
# Check artifacts directory
ls artifacts/
# Should show: feature_spec.json, model_config.json, operating_points.json, train_fps.npz

ls artifacts/models/
# Should show: ensemble_seed1-5.joblib and isotonic.joblib
```

## Using with Streamlit App

Once you have the artifacts:

1. **Local deployment:**
   - Make sure `artifacts/` directory is in the same directory as `streamlit_app.py`
   - Run: `streamlit run streamlit_app.py`

2. **Streamlit Cloud deployment:**
   - Upload the `artifacts/` directory to your GitHub repository
   - Make sure it's in the root directory (same level as `streamlit_app.py`)
   - Deploy to Streamlit Cloud - it will automatically find the artifacts

## Important Notes

- **Without artifacts**: The app will still work for descriptor computation and SMILES validation
- **With artifacts**: Full BBB permeability predictions are available
- **File sizes**: Model files can be several MB each - consider Git LFS for large repositories

## Need Help?

- Check the original manuscript for details on the training data
- Contact: Dr. Sivanesan Dakshanamurthy - sd233@georgetown.edu
- Review the training code in the original Colab notebook
