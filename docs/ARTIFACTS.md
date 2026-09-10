# Artifact inventory

## Models

`artifacts/models/` contains 72 discretized DDLGNs:

- `mnist/`: 24 models;
- `fashion-mnist/`: 24 models; and
- `uci-phishing/`: 24 models.

Each dataset covers depths 1--6 and widths 2K, 4K, 6K, and 8K. Every model directory contains:

- `best_lgn_gates.csv`: ordered gate layer, neuron, input indices, gate ID, and gate name;
- `best_lgn_metadata.json`: architecture, preprocessing, temperature, accuracy, and training provenance; and
- `test_binarized.csv`: the first five preprocessed inputs used by the packaged timing protocol.

The exported gates—not the Python pickle—are the deployed model representation used by both the static analyzer and Rust evaluator.

## Measurements

- `pbs_summary.csv`: per-model accuracy and static PBS totals for the complete grid.
- `pbs_dataset_summary.csv`: dataset-level totals and mean/max bypass rates.
- `selected_models.csv`: Small (2x4K), Medium (4x6K), and Large (6x8K) rows used in Table 3.
- `encrypted_timings.csv`: Boolean and optimized TFHE wall-clock measurements, five inputs per model.
- `qat_fcnn_baselines.csv`: aggregate 10-seed QAT-FCNN values used in Table 5.
- `raw/`: the three source comparison reports from which the encrypted timing CSV was assembled.

`artifacts/MANIFEST.sha256` records SHA-256 checksums for the packaged models and measurements.

## QAT-FCNN baseline code

`baselines/qat-fcnn-mnist/` contains the Python 3.9/Concrete-ML workflow used for the arithmetic MNIST comparison in Table 5. It includes training and FHE evaluation code, a reference ONNX model, compiler MLIR, a 60-run sweep helper, and a JSON-report aggregator. See the baseline README for the environment, commands, and limits of exact reproduction.

## Generated outputs

The committed files under `paper/` are conveniences for paper integration. They can be replaced at any time with `python scripts/reproduce_paper.py`; the canonical numeric inputs live under `artifacts/measurements/`.
