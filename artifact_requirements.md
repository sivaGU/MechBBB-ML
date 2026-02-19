# Artifact Requirements for Tanimoto Similarity Warning

## Required file: `train_fps.npz`

- **Location:** In the same artifacts directory as the model files (e.g. `artifacts/train_fps.npz`).
- **Format:** NumPy compressed archive with one array:
  - **Key:** `"fp"`
  - **Shape:** `(n_training_molecules, 2048)`
  - **Dtype:** `uint8` (0/1), same as ECFP4 bit vector
- **Content:** ECFP4 (Morgan radius=2, n_bits=2048) fingerprints for every molecule in the **BBBP training set** (the same split used to train Model C).

## How it is produced in this project

1. **Precompute script:** `src/16_precompute_train_fp.py`
   - Reads BBBP (or the dataset used for training) training split.
   - Computes Morgan fingerprints with `radius=2`, `n_bits=2048`.
   - Saves to `cache/<dataset>_<split>_train_fp.npz` and optionally copies to `artifacts/train_fps.npz`.

2. **Export script:** `src/41_export_artifacts.py`
   - When building the artifact bundle for the GUI, copies the precomputed `*_train_fp.npz` to `artifacts/train_fps.npz` if it exists.
   - If the precomputed file is missing, the export script reports that `train_fps.npz` is missing and suggests running `16_precompute_train_fp.py`.

## Loading in the GUI

```python
import numpy as np
from pathlib import Path

artifacts_path = Path("artifacts")  # or your configurable path
train_fps = np.load(artifacts_path / "train_fps.npz")["fp"]
# train_fps.shape = (n_train, 2048)
```

## If the GUI repo is separate

- The MechBBB-ML GUI (e.g. MechBBB-ML-GUI repo) must receive `train_fps.npz` as part of the release artifact bundle.
- Fingerprint parameters **must** match the model: **radius=2**, **n_bits=2048** (ECFP4).
- Training set must be the **same** BBBP training split used to train the Model C ensemble (so similarity is "to the model's training chemistry").

## Note

If `train_fps.npz` is not present, the GUI will still function but similarity analysis will be disabled. The app will show an info message: "Training fingerprints not available. Similarity analysis disabled."
