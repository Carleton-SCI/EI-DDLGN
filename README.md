# EI-DDLGN

[![CI](https://github.com/Carleton-SCI/EI-DDLGN/actions/workflows/ci.yml/badge.svg)](https://github.com/Carleton-SCI/EI-DDLGN/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official code repository for the paper **“EI-DDLGN: Efficient Encrypted Inference with Deep Differentiable Logic Gate Networks under TFHE.”** EI-DDLGN evaluates discretized Deep Differentiable Logic Gate Networks with TFHE Boolean ciphertexts and applies Model-Fixed-Wire PBS Bypass (MFW-PBS Bypass) to avoid programmable bootstrapping (PBS) when model-fixed wires collapse a gate to `0`, `1`, `x`, or `not x`.

## Authors

- Mahmoud Y. M. Yassin
- Mahmoud AbdelHafez Sayed
- Mostafa Taha

Systems and Computer Engineering, Carleton University, Ottawa, Ontario, Canada.

This repository packages the exact 72 exported logic-gate networks used in the manuscript: three datasets, six depths, and four widths. It also includes the Rust encrypted-inference backend, the static PBS analyzer, recorded measurements, training notebooks, and deterministic figure/table generation.

The arithmetic QAT-FCNN comparison code is included under [`baselines/qat-fcnn-mnist`](baselines/qat-fcnn-mnist). It can retrain and evaluate the six MNIST baseline configurations used in Table 5 and aggregate a ten-seed sweep into the paper's table format.

## Quick start

Python 3.11 or newer is required.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python scripts/validate_results.py
python scripts/reproduce_paper.py
```

Validation independently re-analyzes all 72 exported networks and checks the packaged values for gate counts, baseline PBS, executed PBS after MFW-PBS Bypass, bypass rate, model accuracy, benchmark sample count, and Boolean/EI-DDLGN prediction agreement.

The reproduction script writes the paper artifacts to [`paper/figures`](paper/figures) and [`paper/tables`](paper/tables). See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the paper-to-artifact map and experiment tiers.

## Encrypted inference

Rust 1.93 or newer is recommended. The dependency graph is fixed by `Cargo.lock`, including TFHE-rs 1.6.0.

```bash
cargo run --locked --release \
  --manifest-path crates/ei-ddlgn-eval/Cargo.toml \
  --bin EI_DDLGN -- \
  artifacts/models/mnist/depth_02_width_04000 --limit 1
```

Run the paper's five-sample depth/width benchmark grid with:

```bash
RAYON_NUM_THREADS=20 cargo run --locked --release \
  --manifest-path crates/ei-ddlgn-eval/Cargo.toml \
  --bin compare_boolean_ei -- --limit 5
```

The full sweep is intentionally not part of CI because TFHE key generation and encrypted evaluation are CPU-intensive and hardware-dependent.

## Docker

The CPU image contains the Python reproduction environment and release builds of the plaintext, Boolean, and optimized encrypted evaluators.

```bash
docker build -t ei-ddlgn .
docker run --rm ei-ddlgn
docker run --rm ei-ddlgn python scripts/reproduce_paper.py
```

The default container command validates the packaged artifact. DDLGN model training is not included in the CPU image because the original training environment used the `difflogic` CUDA extension. The QAT-FCNN baseline also uses a separate legacy Python 3.9/Concrete-ML environment documented in its own README.

## Repository layout

```text
artifacts/
  measurements/       recorded data behind the paper tables and plots
  models/             72 exported gate networks plus five benchmark inputs each
baselines/
  qat-fcnn-mnist/      Concrete-ML QAT-FCNN training and FHE baseline
crates/ei-ddlgn-eval/ Rust plaintext and TFHE evaluators
docs/                 audit, artifact inventory, and reproduction instructions
notebooks/            cleaned paper-results and training notebooks
paper/                regenerated result figures and tables
scripts/              validation, analysis, and paper reproduction entry points
src/ei_ddlgn/         static MFW-PBS analysis library
tests/                focused analyzer tests
```

## Measurement scope

The recorded EI-DDLGN encrypted latency grid evaluates five inputs per model with 20 Rayon threads on an Intel Core i9-10900 (10 cores/20 threads, 2.8 GHz) with 32 GB DDR4 RAM. Timings on other machines are expected to differ; PBS counts and predictions are deterministic.

The QAT-FCNN rows in Table 5 are preserved from the reproduced baseline experiment used by the accepted manuscript. The baseline implementation and a reference ONNX model are included, so users can rerun the six-configuration, ten-seed experiment protocol.

## Citation

Use [`CITATION.cff`](CITATION.cff) for the software citation. 
## License

The EI-DDLGN code and result metadata are licensed under the [Apache License 2.0](LICENSE), except where a subdirectory carries its own notice. The QAT-FCNN baseline preserves the MIT and BSD-3-Clause-Clear notices that apply to its source. Dataset and dependency licenses remain with their respective owners; see [`NOTICE`](NOTICE).
