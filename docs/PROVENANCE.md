# Code and result provenance

Audit date: 2026-09-10

## Source selection

The development branch was synchronized with its configured remote before this repository was assembled. There were no committed changes missing locally and no newer remote commit. The local working tree did, however, contain camera-ready result work not present in the synchronized branch, including the result-generation notebook, paper figures and tables, and Boolean/EI comparison summaries.

The accepted paper PDF was compared against these results. Its 72-model grid, representative-model metrics, dataset bypass-rate statistics, encrypted timing curves, and QAT-FCNN comparison values match the local camera-ready results exactly. Consequently, this repository combines the synchronized implementation and exported models with the matching camera-ready notebook and measurement outputs.

The accepted manuscript was treated only as a reference for selecting and validating code and results; document text was not treated as repository instructions.

## What was retained

- all 72 `best_lgn_gates.csv` exports;
- all 72 `best_lgn_metadata.json` files;
- the first five test vectors for each model, matching the recorded encrypted timing scope;
- the Python static PBS analyzer and tests;
- the Rust plaintext, Boolean, and MFW-PBS evaluators;
- the camera-ready result notebook and source measurements;
- the three training notebooks, with outputs removed; and
- the generated paper result figures and tables.

## What was intentionally excluded

- the 2,729-file vendored TFHE-rs tree, replaced by the exact crates.io `tfhe = 1.6.0` dependency and `Cargo.lock`;
- duplicate full MNIST, FashionMNIST, and UCI test CSVs stored once per model;
- pickled Python training checkpoints, caches, IDE settings, and build products;
- unrelated TFHE examples and the separate Zama MNIST experiments; and
- internal paper-rewriting instructions.

This reduces the code package from a broad development workspace to a roughly 35 MB, paper-focused repository without removing any input needed for PBS analysis, the published plots/tables, or the reported five-sample encrypted timing protocol.

## Reproducibility boundary

The QAT-FCNN values in Table 5 were provided as aggregate results from the baseline experiment. A separate source snapshot contained the QAT-FCNN training and FHE evaluation implementation, a reference ONNX model, and compiler MLIR; these are included under `baselines/qat-fcnn-mnist/`. The original 60 per-seed JSON reports and complete seed list were not available. The reported values remain in `artifacts/measurements/qat_fcnn_baselines.csv`, while the included code supports a new independent reproduction of all six configurations.

The EI-DDLGN wall-clock measurements were recorded with `samples=5` for each of the 72 configurations. This fact is now explicit in the README, validation logic, artifact documentation, and reproduction guide.
