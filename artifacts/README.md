# BBB Model Artifacts

This directory contains the artifacts required for the BBB Permeability Prediction GUI.

## Model Information

- **Dataset:** cornelissen_bbb
- **Split:** scaffold
- **Model:** LGBM_ENSEMBLE5 (5 LightGBM models, averaged)
- **Calibration:** Isotonic regression
- **Thresholds:** Validation-selected operating points

## Artifact Files

### Configuration Files

- `feature_spec.json` - Feature specification (physchem10 + morgan2048 = 2058 features)
- `model_config.json` - Model training metadata
- `operating_points.json` - Validation-selected decision thresholds

### Model Files (`models/`)

- `ensemble_seed1.joblib` through `ensemble_seed5.joblib` - 5 LightGBM ensemble models
- `isotonic.joblib` - Isotonic calibration model

### Similarity Data

- `train_fps.npz` - Precomputed Morgan fingerprints for training set (for similarity checking)

## Feature Specification

The model uses:
- **10 physicochemical descriptors:** MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, HeavyAtomCount, FractionCSP3, NumAromaticRings
- **2048-bit Morgan fingerprints:** radius=2, n_bits=2048
- **Total:** 2058 features

## Usage

These artifacts are used by the Streamlit GUI (`app/app.py`) to make BBB permeability predictions.

### Prediction Workflow

1. Input: SMILES string
2. Standardize SMILES using RDKit
3. Compute features (physchem + Morgan)
4. Ensemble prediction (average of 5 models)
5. Isotonic calibration
6. Apply operating point thresholds
7. Compute similarity to training set
8. Output: Calibrated probability, decisions, similarity warning

## Limitations

1. **Training Data:** Model trained on cornelissen_bbb dataset with scaffold split
2. **Chemical Space:** Predictions are most reliable for molecules similar to training set
3. **Similarity Warning:** Molecules with similarity < 0.3 to training set may have unreliable predictions
4. **Domain:** Model is validated for BBB permeability prediction only

## Validation

- Model validated on scaffold-split test set
- External validation on BBBP dataset
- Calibration verified on validation set
- Operating points selected to optimize MCC and sensitivity

## References

For details on model training, calibration, and validation, see:
- Model training: `src/08_train_models.py`
- Calibration: `src/10_calibrate.py`
- Threshold selection: `src/15_threshold_sweep.py`

## Verification

Run `src/42_verify_artifacts_safe.py` to verify all artifacts are correctly formatted and functional.
