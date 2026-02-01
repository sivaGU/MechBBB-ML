"""
CalcBB prediction module. Two-stage Model C.
Stage-1: efflux, influx, PAMPA (PhysChem + ECFP4)
Stage-2: PhysChem + ECFP4 + mechanistic probs yields BBB+
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs


@dataclass
class PredictResult:
    """Result of a single prediction."""
    is_valid: bool
    smiles: str
    canonical_smiles: str
    prob: float
    bbb_class: str
    p_efflux: Optional[float]
    p_influx: Optional[float]
    p_pampa: Optional[float]
    threshold: float
    error: str


def _compute_features_2058(smiles: str) -> Optional[np.ndarray]:
    """Compute PhysChem (10) + ECFP4 (2048) = 2058 features for stage1."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    phys = np.array([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.HeavyAtomCount(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
    ], dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return np.hstack([phys, arr])


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


class CalcBBPredictor:
    """Loaded predictor state (stage1 + stage2 models, threshold)."""

    def __init__(
        self,
        stage1_efflux,
        stage1_influx,
        stage1_pampa,
        stage2_models: List,
        threshold: float,
    ):
        self.stage1_efflux = stage1_efflux
        self.stage1_influx = stage1_influx
        self.stage1_pampa = stage1_pampa
        self.stage2_models = stage2_models
        self.threshold = threshold

    def predict(self, smiles: str) -> PredictResult:
        """Run full pipeline for one SMILES."""
        canon = _canonicalize_smiles(smiles)
        if canon is None:
            return PredictResult(
                is_valid=False,
                smiles=smiles,
                canonical_smiles="",
                prob=0.0,
                bbb_class="BBB-",
                p_efflux=None,
                p_influx=None,
                p_pampa=None,
                threshold=self.threshold,
                error="Invalid SMILES",
            )
        feats_2058 = _compute_features_2058(canon)
        if feats_2058 is None:
            return PredictResult(
                is_valid=False,
                smiles=smiles,
                canonical_smiles=canon,
                prob=0.0,
                bbb_class="BBB-",
                p_efflux=None,
                p_influx=None,
                p_pampa=None,
                threshold=self.threshold,
                error="Could not compute features",
            )
        X1 = feats_2058.reshape(1, -1)
        p_efflux = float(self.stage1_efflux.predict(X1)[0])
        p_influx = float(self.stage1_influx.predict(X1)[0])
        p_pampa = float(self.stage1_pampa.predict(X1)[0])
        feats_2061 = np.hstack([feats_2058, [p_efflux, p_influx, p_pampa]])
        X2 = feats_2061.reshape(1, -1)
        probs = []
        for m in self.stage2_models:
            p = m.predict_proba(X2)[:, 1][0]
            probs.append(float(p))
        prob = float(np.mean(probs))
        bbb_class = "BBB+" if prob >= self.threshold else "BBB-"
        return PredictResult(
            is_valid=True,
            smiles=smiles,
            canonical_smiles=canon,
            prob=prob,
            bbb_class=bbb_class,
            p_efflux=p_efflux,
            p_influx=p_influx,
            p_pampa=p_pampa,
            threshold=self.threshold,
            error="",
        )


def load_predictor(artifact_dir: Union[str, Path]) -> CalcBBPredictor:
    """Load CalcBB predictor from artifact directory."""
    base = Path(artifact_dir)
    art = base / "artifacts"
    if not art.exists():
        art = base
    stage1_efflux = joblib.load(art / "stage1_efflux.joblib")
    stage1_influx = joblib.load(art / "stage1_influx.joblib")
    stage1_pampa = joblib.load(art / "stage1_pampa.joblib")
    stage2_models = []
    stage2_dir = art / "stage2_modelC"
    for i in range(5):
        stage2_models.append(joblib.load(stage2_dir / f"model_seed{i}.pkl"))
    thresh_data = {}
    thresh_path = art / "threshold.json"
    if thresh_path.exists():
        import json
        thresh_data = json.load(open(thresh_path, "r", encoding="utf-8"))
    threshold = float(thresh_data.get("threshold", 0.35))
    return CalcBBPredictor(
        stage1_efflux=stage1_efflux,
        stage1_influx=stage1_influx,
        stage1_pampa=stage1_pampa,
        stage2_models=stage2_models,
        threshold=threshold,
    )


def predict_single(
    smiles: str,
    threshold: Optional[float] = None,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[CalcBBPredictor] = None,
) -> PredictResult:
    """Predict for a single SMILES."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    if threshold is not None:
        predictor.threshold = threshold
    return predictor.predict(smiles)


def predict_batch(
    smiles_list: List[str],
    threshold: Optional[float] = None,
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[CalcBBPredictor] = None,
) -> List[PredictResult]:
    """Predict for a list of SMILES."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    if threshold is not None:
        predictor.threshold = threshold
    return [predictor.predict(s) for s in smiles_list]
