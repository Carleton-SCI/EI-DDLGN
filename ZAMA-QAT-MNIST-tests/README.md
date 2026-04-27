# QAT-FCNN MNIST 

This folder contains a local MNIST experiment workflow adapted from Zama's Concrete-ML example. It supports quantization-aware model use and FHE/VL evaluation, then saves structured JSON reports under `experiment_reports/`.

## 1) Environment Setup

Run these commands from this directory (`use_case_examples/mnist`).

### One-time setup

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
mkdir -p .checkpoints
```

### For each new terminal

```bash
source .venv/bin/activate
python mnist_in_fhe.py
```

## 2) How To Run Experiments

The script currently uses hardcoded parameters in `mnist_in_fhe.py` and quantization range settings in `model.py`.

### Main experiment knobs (`mnist_in_fhe.py`)

- `epochs`
- `sparsity`
- `quantization_bits`
- `do_test_in_fhe`
- `do_training`
- `show_mlir`
- `lr`
- `gamma`
- `test_data_length_reduced`
- `test_data_length_full`
- `batch_size`
- `test_batch_size`
- `use_cuda_if_available`
- `seed`

### Quantization range knob (`model.py`)

- `CommonQuant.narrow_range`
  - `True` -> report names include `_nr1_`
  - `False` -> report names include `_nr0_`

### Important behavior

- If `do_training = False`, the script loads `mnist.qat.onnx` from the current folder.
- A new JSON report is written to `experiment_reports/` after each successful run.
- The script also snapshots the ONNX model into `experiment_reports/` with the same base name.

## 3) How To Navigate Experiment Outputs

All reports are in:

```text
experiment_reports/
```

Each run creates:

- `<base_name>.json`: settings + metrics
- `<base_name>.onnx`: model snapshot (if `mnist.qat.onnx` exists)

The filename pattern is:

```text
mnist_q{quantization_bits}_s{sparsity}_seed{seed}_nr{0|1}_fhe{test_data_length_reduced}_{timestamp}.json
```

Example:

```text
mnist_q2_s11_seed3108559580_nr1_fhe5_20260226_175713.json
```

## 4) How To Read The Report JSON

Top-level fields:

- `timestamp`: run timestamp (`YYYYMMDD_HHMMSS`)
- `settings`: all runtime parameters used in that run
- `results`: accuracy, bit width, and timing outputs
- `model_path`: source ONNX used by the run
- `model_snapshot_path`: copied ONNX snapshot for reproducibility

### `results.accuracy`

- `VL full`: Virtual Library accuracy on the full test set (`test_data_length_full`)
- `VL short`: Virtual Library accuracy on reduced test size (`test_data_length_reduced`)
- `FHE short`: FHE accuracy on reduced test size

### `results.bit_widths`

- Maximum integer bit-width observed in the compiled graph for each mode.
- Values above 8 are rejected by the script.

### `results.timing`

- `fhe_encrypt_run_decrypt_avg_seconds`: average per-sample FHE time.
- `fhe_samples`: number of samples used for this timing average.
- This timing measures `encrypt_run_decrypt` only; it does not include training and is separate from key generation/compile stage timing in the current reporting.

## 5) Reproducibility And Hardware Context

The experiments documented here were run on CPU:

- CPU: Intel Core i9-10900
- Cores / Threads: 10 / 20
- Base Frequency: 2.8 GHz
- Max Turbo Frequency: 5.2 GHz
- Cache: 20 MB L3
- Architecture: x86-64
- Virtualization: VT-x supported

RAM:

- Total Installed: 32 GB
- Type: DDR4 (SK Hynix)
- Rated Speed: 3200 MT/s
- Configured Speed: 2933 MT/s
- Slots Used: 1 of 4
- Maximum Supported: 128 GB

## 6) Source Attribution

- Core code basis: Zama Concrete-ML example repository  
  `https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/use_case_examples`
- Related paper: *Deep Neural Networks for Encrypted Inference with TFHE*
