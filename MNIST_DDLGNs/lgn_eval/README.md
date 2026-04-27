# MNIST DDLGN Evaluation in Rust

## Overview
This crate (`lgn_eval`) provides a Rust-based implementation for evaluating Differentiable Logic Gate Networks (DDLGNs) on the MNIST dataset.

It supports:
- plaintext evaluation, for correctness checks and baseline timing
- encrypted evaluation using **Fully Homomorphic Encryption (FHE)** via [TFHE-rs](https://github.com/zama-ai/tfhe-rs)

In the encrypted setting, inference is performed without exposing the raw input images to the server.

## Components
The project contains one plaintext evaluator and three encrypted evaluators:

1. **Plaintext Evaluator** (`src/main.rs`):
   * **Execution**: `cargo run --release -- <MODEL_DIR>`
   * **Logic**: Evaluates the exported gate network directly on clear boolean inputs.
   * **Use Case**: Fast reference path to verify exported models, compute baseline accuracy, and compare with encrypted results.

2. **Boolean Backend** (`bin/lgn_eval_boolean`):
    *   **Implementation**: Uses `tfhe::boolean`.
    *   **Logic**: Native simulation of logic gates (AND, OR, XOR, NOT, NAND, NOR, XNOR).
    *   **Use Case**: Typically the most efficient for pure boolean logic networks.

3. **Shortint Backend** (`bin/lgn_eval_shortint`):
    *   **Implementation**: Uses `tfhe::shortint`.
    *   **Logic**: Represents booleans as encrypted 2-bit integers (values 0 and 1). Gates are evaluated using **Programmable Bootstrapping (PBS)** and Bivariate Look-Up Tables (LUTs).
    *   **Use Case**: Demonstrates the flexibility of PBS to implement arbitrary functions.

4. **Integer Backend** (`bin/lgn_eval_integer`):
    *   **Implementation**: Uses `tfhe::integer` (Radix decomposition).
    *   **Logic**: Treat inputs as encrypted integers but utilizes efficient bitwise operations (`unchecked_bitand`, `bitnot`, etc.) provided by the integer crate.
    *   **Use Case**: Suitable when integrating boolean logic into larger arithmetic calculations.

## Prerequisites
*   **Rust**: Stable toolchain (install via `rustup`).
*   **TFHE-rs**: Automatically handled by Cargo dependencies.
*   **Model & Data**: The binaries expect a directory containing:
    *   `best_lgn_gates.csv`: The network topology definition.
    *   `best_lgn_metadata.json`: Model metadata (input dimensions, class counts).
    *   `test_binarized.csv`: The binarized MNIST test dataset (labels + boolean pixels).

## Building
To build the project in release mode (critical for FHE performance):

```bash
cd lgn_eval
cargo build --release
```
(Note: the first command "cd lgn_eval" changes the working directory to inside the rust project, i.e. inside `lgn_eval`. All following commands assume lgn_eval to be the working directory. If you wish to run them from a higher level, you need to include the path to the manifest, as shown in `PLAINTEXT_EVALUATION_GUIDE.md`.)

## Usage

### Plaintext Evaluator

```bash
cargo run --release -- <MODEL_DIR>
```

Example:

```bash
cargo run --release -- ../trained_models/20260309_220106
```

This path evaluates the exported LGN without encryption and prints the final accuracy on `test_binarized.csv`.

### Command Line Interface
All three encrypted binaries share the same CLI argument structure:

```bash
cargo run --release --bin <BINARY_NAME> -- <MODEL_DIR> [--limit N]
```

*   `<BINARY_NAME>`: One of `lgn_eval_boolean`, `lgn_eval_shortint`, `lgn_eval_integer`.
*   `<MODEL_DIR>`: Path to the directory containing the model and data files.
*   `--limit N` (Optional): Limits the evaluation to the first `N` test vectors. Highly recommended for quick benchmarks as FHE is computationally intensive.

### Parallelization (IMPORTANT)
The inference engine utilizes **Rayon** for parallel processing to speed up evaluation.
*   **Strategy**: Neurons within the same layer are evaluated in parallel.
*   **Configuration**: Parallelism is enabled by default in the source code (`USE_PARALLEL = true`) of each bin file (lgn_eval_boolean, lgn_eval_shortint, lgn_eval_integer)
*   **Thread Control**: You can control the number of threads used by setting the `RAYON_NUM_THREADS` environment variable.

```bash
# Example: Run with 16 threads
RAYON_NUM_THREADS=16 cargo run --release --bin lgn_eval_boolean -- ...
```

## Examples

**1. Run plaintext evaluation**
```bash
cargo run --release -- ../trained_models/20260309_220106
```

**2. Run Boolean Backend**
```bash
cargo run --release --bin lgn_eval_boolean -- ../trained_models/20260207_121201 --limit 5
```

**3. Run Shortint Backend**
```bash
cargo run --release --bin lgn_eval_shortint -- ../trained_models/20260207_121201 --limit 5
```

**4. Run Integer Backend**
```bash
cargo run --release --bin lgn_eval_integer -- ../trained_models/20260207_121201 --limit 5
```

## Output Interpretation
The tools provide a step-by-step log and final statistics.

For the plaintext evaluator:
- **Accuracy**: Ratio of correctly predicted labels over the processed test set.

For the encrypted evaluators:
- **Accuracy**: Ratio of correctly predicted labels over the processed test set.
- **Total encryption time**: Cumulative time spent encrypting all evaluated input vectors.
- **Total evaluation time**: Cumulative time spent evaluating the encrypted network.
- **Total decryption time**: Cumulative time spent decrypting all output vectors.
- **Avg encrypt/vector**: Average encryption time per image.
- **Avg eval/vector**: Average encrypted evaluation time per image.
- **Avg decrypt/vector**: Average decryption time per image.
- **Avg end-to-end/vector**: Average total time per image computed as `encrypt + eval + decrypt`.

`Avg end-to-end/vector` is the main end-user latency metric for encrypted inference.
