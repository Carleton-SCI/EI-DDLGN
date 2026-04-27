# EI-DDLGN: Efficient Encrypted Inference with Deep Differential Logic Gate Networks under TFHE

This repository contains code and experiments aligned with our paper:

"EI-DDLGN: Efficient Encrypted Inference with Deep Differential Logic Gate Networks under TFHE"

The project focuses on privacy-preserving MNIST inference using Deep Differential Logic Gate Networks (DDLGNs) and a Boolean-native TFHE evaluation backend.

## Scope and Main Idea

- DDLGNs are trained and discretized into logic-gate circuits.
- Encrypted inference is executed as Boolean gate evaluation under TFHE.
- This avoids integer-accumulation sensitivity to circuit bit-width that appears in QAT-based arithmetic baselines.

## Repository Layout

- `EI-DDLGN Framework/`: Core EI-DDLGN artifacts for MNIST.
    - `DDLGNs__MNIST.ipynb`: notebook for training/export workflow.
    - `trained_models/`: exported models and binarized test sets.
    - `lgn_eval/`: Rust evaluator crate for plaintext checks and encrypted Boolean inference.
- `ZAMA-QAT-MNIST-tests/`: baseline workflow used for QAT-FCNN comparison.
- `tfhe-rs/`: TFHE-rs codebase used as cryptographic backend.

For evaluator usage details, see `EI-DDLGN Framework/lgn_eval/README.md`.

## Reported Results

For EI-DDLGN on MNIST (Boolean backend):

- Small model: 92.14% accuracy, 6.995 s/image total latency.
- Medium model: 96.11% accuracy, 20.164 s/image total latency.
- Large model: 97.23% accuracy, 33.731 s/image total latency.

Compared to reproduced QAT-FCNN baselines in the same paper:

- FCNN accuracy range: 91.54% to 92.58%.
- FCNN latency range: 88.24 to 207.98 s/image.

## Quick Start (Boolean Inference)

```bash
cd "EI-DDLGN Framework/lgn_eval"
cargo build --release
cargo run --release --bin lgn_eval_boolean -- ../trained_models/20260309_220106 --limit 5
```

## Notes and Limitations

As reported in the paper, evaluation is currently centered on feedforward DDLGNs and MNIST, with one-image-at-a-time encrypted inference under an honest-but-curious threat model.
