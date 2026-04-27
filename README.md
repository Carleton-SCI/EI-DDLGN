# PP-DDLGN: Privacy Preserving Differentiable Logic Gate Networks

This repository implements Privacy Preserving Differentiable Logic Gate Networks (DDLGNs), focusing on encrypted inference for the MNIST dataset using Fully Homomorphic Encryption (FHE).

## Project Structure

The repository is organized as follows:

- **`MNIST_DDLGNs/`**: Main directory containing the implementation and resources for MNIST DDLGNs.
  - **`lgn_eval/`**: A Rust crate for evaluating DDLGNs on encrypted data. It supports multiple TFHE backends (Boolean, Shortint, and Integer).
  - **`trained_models/`**: Contains exported model architectures (`best_lgn_gates.csv`), metadata, and binarized test datasets.
  - **`DDLGNs__MNIST.ipynb`**: Jupyter notebook for training the DDLGN models and performing initial binarization.
  - **`GUIDE_FOR_MAHMOUD.md`**: A detailed guide for reconstructing and validating the exported models.
  - **`DiffLogic_Installation_Guide.pdf`**: Documentation for setting up the environment.
  - **`ref_main.py` & `ref_mnist_dataset.py`**: Reference Python scripts for model handling.
- **`tfhe-rs/`**: Zama's variant of the TFHE Fully Homomorphic Encryption library (Version 1.6.0). This library provides the cryptographic primitives used for encrypted inference.

---

## Encrypted Inference for MNIST DDLGNs
(Please refer to the [lgn_eval/README.md](MNIST_DDLGNs/lgn_eval/README.md) for detailed instructions on how to use the crate.)


### Overview

The `lgn_eval` crate provides a Rust-based implementation for evaluating DDLGNs on the MNIST dataset using **Fully Homomorphic Encryption (FHE)**. It utilizes the [TFHE-rs](https://github.com/zama-ai/tfhe-rs) library to perform inference on encrypted data, ensuring that the input images remain confidential during processing.

### Components
The project implements the same logic gate network evaluation using three different TFHE backends to compare performance and implementation strategies:

1.  **Boolean Backend** (`bin/lgn_eval_boolean`):
    *   **Implementation**: Uses `tfhe::boolean`.
    *   **Logic**: Native simulation of logic gates (AND, OR, XOR, NOT, NAND, NOR, XNOR).
    *   **Use Case**: Typically the most efficient for pure boolean logic networks.

2.  **Shortint Backend** (`bin/lgn_eval_shortint`):
    *   **Implementation**: Uses `tfhe::shortint`.
    *   **Logic**: Represents booleans as encrypted 2-bit integers (values 0 and 1). Gates are evaluated using **Programmable Bootstrapping (PBS)** and Bivariate Look-Up Tables (LUTs).
    *   **Use Case**: Demonstrates the flexibility of PBS to implement arbitrary functions.

3.  **Integer Backend** (`bin/lgn_eval_integer`):
    *   **Implementation**: Uses `tfhe::integer` (Radix decomposition).
    *   **Logic**: Treat inputs as encrypted integers but utilizes efficient bitwise operations (`unchecked_bitand`, `bitnot`, etc.) provided by the integer crate.
    *   **Use Case**: Suitable when integrating boolean logic into larger arithmetic calculations.

### Prerequisites
*   **Rust**: Stable toolchain (install via `rustup`).
*   **TFHE-rs**: Automatically handled by Cargo dependencies.
*   **Model & Data**: The binaries expect a directory containing:
    *   `best_lgn_gates.csv`: The network topology definition.
    *   `best_lgn_metadata.json`: Model metadata (input dimensions, class counts).
    *   `test_binarized.csv`: The binarized MNIST test dataset (labels + boolean pixels).

### Building
To build the project in release mode (critical for FHE performance):

```bash
cd MNIST_DDLGNs/lgn_eval
cargo build --release
```

### Usage

#### Command Line Interface
All three binaries share the same CLI argument structure:

```bash
cargo run --release --bin <BINARY_NAME> -- <MODEL_DIR> [--limit N]
```

*   `<BINARY_NAME>`: One of `lgn_eval_boolean`, `lgn_eval_shortint`, `lgn_eval_integer`.
*   `<MODEL_DIR>`: Path to the directory containing the model and data files.
*   `--limit N` (Optional): Limits the evaluation to the first `N` test vectors.

#### Parallelization
The inference engine utilizes **Rayon** for parallel processing. Neurons within the same layer are evaluated in parallel. You can control the thread count via the `RAYON_NUM_THREADS` environment variable.

### Performance & Accuracy
The tools provide detailed logs including accuracy, total encryption time, total evaluation time, and average latency per vector. For more detailed information on the Rust implementation, refer to the [lgn_eval/README.md](MNIST_DDLGNs/lgn_eval/README.md).
