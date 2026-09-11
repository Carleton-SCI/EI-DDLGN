# QAT-FCNN MNIST baseline

This directory contains the code used to reproduce the arithmetic QAT-FCNN baseline experiments reported in **"EI-DDLGN: Efficient Encrypted Inference with Deep Differentiable Logic Gate Networks under TFHE."** It trains two-bit, narrow-range, quantization-aware fully connected networks on MNIST and evaluates them with Concrete-ML's virtual and FHE backends.

## Paper configurations

The paper reports ten independently trained models for each configuration:

| Configuration | Sparsity multiplier | Active connections per neuron |
|---|---:|---:|
| QAT-FCNN-4 | 4 | 56 |
| QAT-FCNN-6 | 6 | 84 |
| QAT-FCNN-8 | 8 | 112 |
| QAT-FCNN-10 | 10 | 140 |
| QAT-FCNN-11 | 11 | 154 |
| QAT-FCNN-12 | 12 | 168 |

`model.py` defines the network and quantizers. `mnist_in_fhe.py` trains or loads a model, compiles it, checks virtual/FHE predictions, measures `encrypt_run_decrypt`, and writes a structured JSON report. `run_paper_sweep.py` runs the complete six-configuration, ten-seed protocol, and `aggregate_reports.py` converts those reports into Table 5's schema.

The supplied `mnist.qat.onnx` and `mnist.mlir.txt` are retained reference outputs. The ONNX file is a real model, not a Git LFS pointer.

## Reproducibility scope

This code can be used to rerun the QAT-FCNN experiment protocol and reproduce the QAT baseline results. The source directory did not contain the original 60 per-seed JSON reports or the complete original seed list. Therefore:

- `../../artifacts/measurements/qat_fcnn_baselines.csv` remains the canonical copy of the aggregate values reported in the paper;
- new ten-seed sweeps provide an independent empirical reproduction; and
- exact wall-clock latency will depend on the machine, operating system, and Concrete compiler environment.

## Environment

Use Linux x86-64 with Python 3.9. The recorded environment used Python 3.9.25, Concrete-ML 0.3.0, Concrete-Numpy 0.7.0, Concrete-Compiler 0.10.0, PyTorch 1.12.1, and Brevitas 0.7.1.

The old Concrete compiler toolchain does not reliably quote linker paths. Place the environment and checkout in a path without spaces.

```bash
cd baselines/qat-fcnn-mnist
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
```

MNIST is downloaded automatically into the ignored `data/` directory.

## Run one experiment

Train and evaluate QAT-FCNN-4 with the recorded default seed:

```bash
python mnist_in_fhe.py \
  --sparsity 4 \
  --seed 3108559580 \
  --epochs 20 \
  --fhe-samples 5
```

Evaluate the supplied ONNX model without retraining:

```bash
python mnist_in_fhe.py --no-training --fhe-samples 5
```

Each successful run writes a JSON report and an ONNX snapshot under `experiment_reports/`.

## Run and aggregate the paper grid

The following illustrates a deterministic ten-seed sweep. These are illustrative seeds because the complete original seed list was not available.

```bash
python run_paper_sweep.py --seeds \
  3108559580 3108559581 3108559582 3108559583 3108559584 \
  3108559585 3108559586 3108559587 3108559588 3108559589

python aggregate_reports.py experiment_reports \
  --output qat_fcnn_aggregated.csv
```

The full sweep performs 60 training, compilation, and FHE runs and can take many hours. Use `run_paper_sweep.py ... --dry-run` to inspect all commands first.

## Measurement interpretation

- `accuracy.VL full` is virtual-library accuracy on the full MNIST test set.
- `accuracy.FHE short` is FHE accuracy on the reduced timing subset.
- `bit_widths.FHE short` is the compiled circuit's maximum integer bit width.
- `timing.fhe_encrypt_run_decrypt_avg_seconds` is the mean per-image FHE time and excludes training, compilation, and key generation.

## licenses

The local baseline modifications are provided under the included MIT license. The core example was adapted from [Zama Concrete-ML's MNIST example](https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/use_case_examples/mnist), whose BSD-3-Clause-Clear notice is also included. Installed dependencies and MNIST remain governed by their respective terms.
