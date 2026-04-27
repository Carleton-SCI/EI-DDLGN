# EI-DDLGN MNIST Evaluation in Rust (Boolean Backend)

## Overview

This crate provides the Rust side of the EI-DDLGN workflow described in the paper,
with a focus on:

- plaintext evaluation for correctness checks
- encrypted Boolean inference under TFHE for privacy-preserving prediction

The encrypted path uses TFHE-rs and evaluates logic-gate networks directly as Boolean operations.

## Components

1. Plaintext evaluator (`src/main.rs`)
- Execution: `cargo run --release -- <MODEL_DIR>`
- Purpose: verifies exported gate networks and gives a clear-text reference.

2. Boolean encrypted evaluator (`src/bin/lgn_eval_boolean.rs`)
- Execution: `cargo run --release --bin lgn_eval_boolean -- <MODEL_DIR> [--limit N]`
- Purpose: evaluates the exported DDLGN using TFHE Boolean ciphertexts.

## Prerequisites

- Rust stable toolchain (via `rustup`)
- Cargo dependencies (pulled automatically)
- A model directory containing:
  - `best_lgn_gates.csv`
  - `best_lgn_metadata.json`
  - `test_binarized.csv`

## Build

```bash
cd lgn_eval
cargo build --release
```

## Usage

### Plaintext

```bash
cargo run --release -- ../trained_models/20260309_220106
```

### Encrypted Boolean

```bash
cargo run --release --bin lgn_eval_boolean -- ../trained_models/20260309_220106 --limit 5
```

Optional:

- `--limit N` evaluates only the first `N` vectors for quick tests.

## Parallelization

Neuron evaluation within a layer is parallelized with Rayon.
Control thread count with `RAYON_NUM_THREADS`:

```bash
RAYON_NUM_THREADS=16 cargo run --release --bin lgn_eval_boolean -- ../trained_models/20260309_220106 --limit 5
```

## Output Metrics

Encrypted runs report:

- accuracy
- total encryption time
- total evaluation time
- total decryption time
- average encrypt/eval/decrypt time per image
- average end-to-end latency per image

In the paper's experiments, homomorphic evaluation is the dominant latency component.
