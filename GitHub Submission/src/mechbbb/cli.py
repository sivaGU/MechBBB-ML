"""
MechBBB CLI — predict from command line.
Usage:
  python -m src.mechbbb.cli --smiles "CCO" "c1ccccc1" --output out.csv --artifact-dir .
  python -m src.mechbbb.cli --input example_inputs.csv --output out.csv --artifact-dir .
"""
import argparse
import csv
import sys
from pathlib import Path

from .predict import load_predictor, predict_batch, predict_single


def main():
    parser = argparse.ArgumentParser(description="MechBBB BBB permeability predictor")
    parser.add_argument("--smiles", nargs="+", help="SMILES strings to predict")
    parser.add_argument("--input", help="Input CSV file (must have SMILES column)")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--artifact-dir", default=".", help="Path to artifact directory")
    parser.add_argument("--threshold", type=float, default=0.35, help="Classification threshold")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    if not (artifact_dir / "artifacts").exists() and not (artifact_dir / "stage1_efflux.joblib").exists():
        art_check = artifact_dir / "artifacts"
        if not art_check.exists():
            sys.stderr.write(f"Error: artifacts not found in {artifact_dir}\n")
            sys.stderr.write("Expected artifacts/ with stage1_efflux.joblib, stage1_influx.joblib, etc.\n")
            sys.exit(1)

    predictor = load_predictor(artifact_dir)
    predictor.threshold = args.threshold

    if args.smiles:
        results = predict_batch(args.smiles, predictor=predictor)
        rows = []
        for r in results:
            rows.append({
                "smiles": r.smiles,
                "canonical_smiles": r.canonical_smiles,
                "prob_BBB+": f"{r.prob:.6f}" if r.is_valid else "",
                "label": 1 if (r.is_valid and r.bbb_class == "BBB+") else (0 if r.is_valid else ""),
                "BBB_class": r.bbb_class if r.is_valid else "",
                "p_efflux": f"{r.p_efflux:.6f}" if r.p_efflux is not None else "",
                "p_influx": f"{r.p_influx:.6f}" if r.p_influx is not None else "",
                "p_pampa": f"{r.p_pampa:.6f}" if r.p_pampa is not None else "",
                "threshold": f"{r.threshold:.2f}" if r.is_valid else "",
                "error": r.error,
            })
    elif args.input:
        inp = Path(args.input)
        if not inp.exists():
            sys.stderr.write(f"Error: input file not found: {inp}\n")
            sys.exit(1)
        import pandas as pd
        df = pd.read_csv(inp)
        col = next((c for c in df.columns if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"), None)
        if col is None:
            sys.stderr.write(f"Error: CSV must have SMILES column. Found: {list(df.columns)}\n")
            sys.exit(1)
        smiles_list = df[col].astype(str).tolist()
        results = predict_batch(smiles_list, predictor=predictor)
        rows = []
        for r in results:
            rows.append({
                "smiles": r.smiles,
                "canonical_smiles": r.canonical_smiles,
                "prob_BBB+": f"{r.prob:.6f}" if r.is_valid else "",
                "label": 1 if (r.is_valid and r.bbb_class == "BBB+") else (0 if r.is_valid else ""),
                "BBB_class": r.bbb_class if r.is_valid else "",
                "p_efflux": f"{r.p_efflux:.6f}" if r.p_efflux is not None else "",
                "p_influx": f"{r.p_influx:.6f}" if r.p_influx is not None else "",
                "p_pampa": f"{r.p_pampa:.6f}" if r.p_pampa is not None else "",
                "threshold": f"{r.threshold:.2f}" if r.is_valid else "",
                "error": r.error,
            })
    else:
        parser.print_help()
        sys.exit(0)

    out_path = args.output
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        import csv as csvmod
        writer = csvmod.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
