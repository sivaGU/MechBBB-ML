"""
ECFP4 Tanimoto similarity to training set — for GUI applicability domain warning.

Use the same Morgan fingerprint as the model: radius=2, n_bits=2048 (ECFP4).
Training fingerprints must be loaded from train_fps.npz: np.load("train_fps.npz")["fp"].
"""

import numpy as np


def compute_morgan(smiles_list, radius=2, n_bits=2048):
    """
    Compute Morgan (ECFP4) fingerprints for a list of SMILES.
    Returns numpy array of shape (n, n_bits) with dtype uint8 (0/1).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n = len(smiles_list)
    fp_array = np.zeros((n, n_bits), dtype=np.uint8)

    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            for j in range(n_bits):
                if fp.GetBit(j):
                    fp_array[i, j] = 1
        except Exception:
            pass

    return fp_array


def compute_similarity(query_fp, train_fps):
    """
    Compute maximum Tanimoto similarity between query fingerprint and training set.

    Tanimoto for binary fingerprints: |A ∩ B| / |A ∪ B|.

    Args:
        query_fp: 1D array of shape (n_bits,) — single molecule fingerprint (0/1).
        train_fps: 2D array of shape (n_train, n_bits) — training set fingerprints.

    Returns:
        float: Maximum Tanimoto similarity in [0, 1].
    """
    if query_fp.sum() == 0:
        return 0.0

    # Vectorized: compare query to every training fingerprint
    intersection = np.logical_and(query_fp[None, :], train_fps).sum(axis=1)
    union = np.logical_or(query_fp[None, :], train_fps).sum(axis=1)
    similarities = np.where(union > 0, intersection / union, 0.0)

    return float(similarities.max())


# Threshold from manuscript: warn when max similarity below 0.30
SIMILARITY_WARNING_THRESHOLD = 0.30


def similarity_flag(max_similarity, threshold=SIMILARITY_WARNING_THRESHOLD):
    """Return 'low' if max_similarity < threshold, else 'ok'."""
    return "low" if max_similarity < threshold else "ok"
