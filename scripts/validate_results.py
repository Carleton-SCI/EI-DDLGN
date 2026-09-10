#!/usr/bin/env python3
"""Validate that packaged models reproduce the paper's recorded PBS results."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ei_ddlgn.pbs_analyzer import analyze_pbs_count  # noqa: E402


MODEL_ROOTS = {
    "MNIST": ROOT / "artifacts" / "models" / "mnist",
    "FashionMNIST": ROOT / "artifacts" / "models" / "fashion-mnist",
    "UCI Phishing Websites": ROOT / "artifacts" / "models" / "uci-phishing",
}


def close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> int:
    recorded = pd.read_csv(ROOT / "artifacts" / "measurements" / "pbs_summary.csv")
    expected_rows = {
        (str(row["Dataset"]), int(row["Depth"]), int(row["Width"])): row
        for _, row in recorded.iterrows()
    }
    errors: list[str] = []
    checked = 0

    manifest_path = ROOT / "artifacts" / "MANIFEST.sha256"
    if not manifest_path.is_file():
        errors.append("Missing artifacts/MANIFEST.sha256")
    else:
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            wanted, relative_path = line.split("  ", maxsplit=1)
            artifact = ROOT / Path(relative_path)
            if not artifact.is_file():
                errors.append(f"Manifest entry is missing: {relative_path}")
                continue
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual.lower() != wanted.lower():
                errors.append(f"Checksum mismatch: {relative_path}")

    for dataset, model_root in MODEL_ROOTS.items():
        model_dirs = sorted(path for path in model_root.iterdir() if path.is_dir())
        if len(model_dirs) != 24:
            errors.append(f"{dataset}: expected 24 model directories, found {len(model_dirs)}")
        for model_dir in model_dirs:
            required = [
                model_dir / "best_lgn_gates.csv",
                model_dir / "best_lgn_metadata.json",
                model_dir / "test_binarized.csv",
            ]
            missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
            if missing:
                errors.append(f"{dataset}/{model_dir.name}: missing {missing}")
                continue

            metadata = json.loads(required[1].read_text(encoding="utf-8"))
            depth = int(metadata["num_layers"])
            width = int(metadata["num_neurons"])
            key = (dataset, depth, width)
            expected = expected_rows.get(key)
            if expected is None:
                errors.append(f"Unexpected model {key}")
                continue

            with required[2].open(newline="", encoding="utf-8-sig") as handle:
                sample_count = sum(1 for _ in csv.reader(handle)) - 1
            if sample_count != 5:
                errors.append(f"{key}: expected 5 packaged benchmark samples, found {sample_count}")

            analysis = analyze_pbs_count(model_dir, verbose=False)
            totals = analysis["totals"]
            comparisons = {
                "total gates": (totals["total_gates"], expected["Total Gates"]),
                "baseline PBS": (totals["total_baseline_pbs"], expected["Baseline PBS"]),
                "optimized PBS": (totals["total_optimized_pbs"], expected["Optimized PBS"]),
                "PBS saved": (totals["total_saved"], expected["PBS Saved"]),
                "PBS reduction": (totals["pbs_saving_percent"], expected["PBS Reduction %"]),
                "accuracy": (
                    analysis["model_metadata"]["test_accuracy_percent"],
                    expected["Test Accuracy %"],
                ),
            }
            for label, (actual, wanted) in comparisons.items():
                if not close(actual, wanted):
                    errors.append(f"{key}: {label} is {actual}, expected {wanted}")
            checked += 1

    timings = pd.read_csv(ROOT / "artifacts" / "measurements" / "encrypted_timings.csv")
    if len(timings) != 72 or not (timings["samples"] == 5).all():
        errors.append("Encrypted timing source must contain 72 rows with five samples each")
    if not close(timings["prediction_match_rate"].min(), 1.0):
        errors.append("Recorded Boolean and EI-DDLGN predictions do not all match")

    selected = pd.read_csv(ROOT / "artifacts" / "measurements" / "selected_models.csv")
    if len(selected) != 9:
        errors.append(f"Selected-model table must contain 9 rows, found {len(selected)}")

    if errors:
        print("Artifact validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Validated {checked} exported models across 3 datasets.")
    print("Recomputed gate totals, baseline PBS, MFW-PBS counts, bypass rates, and recorded accuracies.")
    print("Validated the 72-row, five-sample encrypted timing grid with 100% backend prediction agreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
