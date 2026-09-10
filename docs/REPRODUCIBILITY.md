# Reproducibility guide

The artifact separates deterministic reconstruction from hardware-dependent reruns. This matters because static PBS counts and plots can be reproduced exactly, while encrypted wall-clock latency varies with CPU, thread count, system load, and TFHE build settings.

## Tier 1: validate the packaged paper data

```bash
python -m pip install -e ".[dev]"
python scripts/validate_results.py
```

This re-parses every `best_lgn_gates.csv`, performs layer-wise model-fixed-wire propagation, and compares the result with `artifacts/measurements/pbs_summary.csv`. All 72 models must match for:

- total gates;
- baseline PBS count;
- executed PBS count after MFW-PBS Bypass;
- bypassed PBS count and percentage;
- test accuracy recorded in model metadata;
- the five packaged benchmark inputs; and
- prediction agreement in the recorded Boolean-versus-EI timing grid.

## Tier 2: regenerate the result figures and tables

```bash
python scripts/reproduce_paper.py
```

| Paper artifact | Generated file |
| --- | --- |
| Figure 4: executed PBS share heatmaps | `paper/figures/fps_optimized_pbs_gate_share_heatmaps.{pdf,png}` |
| Figure 5: PBS bypass-rate distribution | `paper/figures/fps_pbs_reduction_distribution.{pdf,png}` |
| Figure 6: accuracy versus executed PBS | `paper/figures/fps_accuracy_vs_optimized_pbs.{pdf,png}` |
| Figure 7: best accuracy under PBS budget | `paper/figures/fps_best_accuracy_under_optimized_pbs_budget.{pdf,png}` |
| Figure 8: encrypted time versus width | `paper/figures/fps_ei_encrypted_time_vs_width.{pdf,png}` |
| Table 3: representative EI-DDLGN models | `paper/tables/fps_selected_models_table.{csv,tex}` |
| Table 5: EI-DDLGN/QAT-FCNN comparison | `paper/tables/fps_arithmetic_comparison_table.{csv,tex}` |

Figures 1--3 and Tables 1--2 are explanatory diagrams/definition tables, not experiment-generated artifacts.

To avoid changing committed outputs during a check, select temporary directories:

```bash
python scripts/reproduce_paper.py --figures-dir build/figures --tables-dir build/tables
```

## Tier 3: rerun encrypted inference

Build the Rust evaluator:

```bash
cargo build --locked --release --manifest-path crates/ei-ddlgn-eval/Cargo.toml
```

Evaluate one representative model:

```bash
cargo run --locked --release \
  --manifest-path crates/ei-ddlgn-eval/Cargo.toml \
  --bin EI_DDLGN -- \
  artifacts/models/mnist/depth_02_width_04000 --limit 5
```

Evaluate both the ordinary Boolean backend and EI-DDLGN over the full 72-model grid:

```bash
RAYON_NUM_THREADS=20 cargo run --locked --release \
  --manifest-path crates/ei-ddlgn-eval/Cargo.toml \
  --bin compare_boolean_ei -- --limit 5 --verbose
```

PowerShell users can set the thread count with `$env:RAYON_NUM_THREADS = "20"` before running the Cargo command.

The packaged `test_binarized.csv` files intentionally contain the same first five vectors used by the recorded timing experiment. Full test sets were duplicated in every model directory in the source repository (roughly 881 MB for the three sweeps) and are not needed to reproduce the paper's timing grid or PBS results.

## Tier 4: retrain the 72 networks

The cleaned source notebooks are under `notebooks/training/`. The original tested environment was:

- Windows 10/11 x64;
- Python 3.10;
- PyTorch 2.5.1 with CUDA 12.1;
- CUDA Toolkit 12.1;
- Visual Studio Build Tools 2022, MSVC 14.38; and
- `difflogic` at commit `81347af` with its CUDA extension enabled.

See `docs/DiffLogic_Installation_Guide.pdf` for the original environment notes. Training is a long GPU workflow and is intentionally separate from the CPU-only reproduction container. The notebooks are preserved as experiment provenance; update their output-root variables when starting a new sweep so new models do not overwrite the packaged artifact.

## Recorded hardware and timing interpretation

The manuscript measurements used an Intel Core i9-10900 with 10 cores/20 threads at 2.8 GHz, 32 GB DDR4 RAM, and 20 Rayon threads. `artifacts/measurements/encrypted_timings.csv` records five samples for every model. Therefore:

- use the recorded CSV to reproduce the published figure and table exactly;
- use reruns to confirm scaling and correctness on a new machine; and
- do not expect wall-clock equality across machines.

## Tier 5: reproduce the QAT-FCNN baseline

`baselines/qat-fcnn-mnist/` contains the Concrete-ML QAT training, compilation, virtual-library evaluation, and FHE evaluation workflow for the six QAT-FCNN rows in Table 5. It also includes a reference ONNX model, the compiler MLIR output, a six-configuration sweep runner, and a report aggregator. Follow its dedicated README in a Linux x86-64 Python 3.9 environment. This legacy dependency stack is intentionally separate from the repository's primary Docker image.

The Table 5 data are aggregate results over 10 independently trained seeds. The implementation is now included, but the original 60 per-seed JSON reports and complete seed list were not available. New runs can independently reproduce the experiment protocol; exact aggregate accuracy and hardware-dependent latency may vary.
