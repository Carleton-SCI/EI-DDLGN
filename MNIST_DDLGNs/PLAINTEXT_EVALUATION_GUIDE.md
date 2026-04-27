# Plaintext Evaluation Guide for MNIST DDLGNs

This guide explains how to run the exported MNIST Logic Gate Networks in Rust without encryption first, validate their accuracy, and prepare the same exported models for later TFHE evaluation.

## Purpose

Use the plaintext Rust evaluator first to confirm that:

- the exported `best_lgn_gates.csv` is valid
- the exported `best_lgn_metadata.json` matches the dataset representation
- the model reaches the expected test accuracy before introducing encryption

This is the fastest correctness check before running the encrypted backends in `lgn_eval`.

## Files You Need

For a given exported run directory, you need:

- `trained_models/<run_id>/best_lgn_gates.csv`
- `trained_models/<run_id>/best_lgn_metadata.json`
- `trained_models/<run_id>/test_binarized.csv`

And from the Rust crate:

- `lgn_eval/Cargo.toml`
- `lgn_eval/src/main.rs`

## Plaintext Rust Evaluator

The plaintext evaluator is:

- `lgn_eval/src/main.rs`

It:

- loads the exported gate network from CSV
- loads model metadata from JSON
- reads `test_binarized.csv`
- evaluates the logic network in cleartext boolean form
- prints final test accuracy

## Build

From `MNIST_DDLGNs`:

```bash
cd lgn_eval
cargo build --release
```

If you prefer running from the parent folder instead:

```bash
cargo build --release --manifest-path lgn_eval/Cargo.toml
```

## Run a Plaintext Model

From inside `lgn_eval`:

```bash
cargo run --release -- ../trained_models/<run_id>
```

From `MNIST_DDLGNs`:

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml -- trained_models/<run_id>
```

The `--` is required so Cargo forwards the model directory to the executable.

## Small, Medium, and Large Models

Your notebook currently defines these model-size presets:

- `small`: `num_neurons=4000`, `num_layers=2`, `tau=3.0`
- `medium`: `num_neurons=6000`, `num_layers=4`, `tau=5.0`
- `large`: `num_neurons=8000`, `num_layers=6`, `tau=10.0`

These settings come from `DDLGNs__MNIST.ipynb`.

To run any exported model in plaintext, point the Rust evaluator to the corresponding run folder.

### Small

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml --bin lgn_eval -- trained_models/20260309_220106
```

### Medium

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml --bin lgn_eval -- trained_models/20260309_224007
```

### Large

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml --bin lgn_eval -- trained_models/20260207_121201
```

Replace `<small_run_id>`, `<medium_run_id>`, and `<large_run_id>` with the actual export folders produced by training.

## How to Identify Which Run Is Small, Medium, or Large

Check `trained_models/<run_id>/log_args.txt`.

Use:

- `4000` neurons and `2` layers -> `small`
- `6000` neurons and `4` layers -> `medium`
- `8000` neurons and `6` layers -> `large`

Example:

```bash
sed -n '1p' trained_models/<run_id>/log_args.txt
```

## Current Example Export Folders

At the moment, the repo contains exported metadata for at least these runs:

- `trained_models/20260309_220106`
- `trained_models/20260207_121201`
- `trained_models/20260309_224007`

Example plaintext runs:

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml --bin lgn_eval -- trained_models/20260309_220106
```

```bash
cargo run --release --manifest-path lgn_eval/Cargo.toml --bin lgn_eval -- trained_models/20260207_121201
```

Before treating a run as `small`, `medium`, or `large`, verify it through `log_args.txt`.

## Metadata Meaning

Open `best_lgn_metadata.json` and use:

- `class_count`: number of output classes
- `in_dim`: flattened binary input dimension
- `num_layers`, `num_neurons`: architecture summary
- `preprocess.binarization`: input binarization rule

Common MNIST cases in this repo:

- `pixel_mode = threshold_05` -> `in_dim = 784`
- `pixel_mode = multi_threshold` with `num_niveis = 3` -> `in_dim = 2352`

## CSV Reconstruction Rule

`best_lgn_gates.csv` contains one row per neuron:

- `layer`
- `neuron`
- `input_a`
- `input_b`
- `gate_id`
- `gate_name`

To reconstruct the network:

1. Process layers in ascending `layer` order.
2. Inside each layer, process neurons in ascending `neuron` order.
3. Compute each neuron output from `gate_name(input_a_value, input_b_value)`.

## Gate Semantics

Use these exact boolean mappings:

- `zero` -> `false`
- `one` -> `true`
- `a` -> `a`
- `b` -> `b`
- `not_a` -> `!a`
- `not_b` -> `!b`
- `and` -> `a & b`
- `or` -> `a | b`
- `xor` -> `a ^ b`
- `not_xor` -> `!(a ^ b)`
- `not_and` -> `!(a & b)`
- `not_or` -> `!(a | b)`
- `implies` -> `(!a) | b`
- `implied_by` -> `a | (!b)`
- `not_implies` -> `a & (!b)`
- `not_implied_by` -> `(!a) & b`

## Final Classification Rule

After the last logic layer:

1. Let output length be `N`.
2. Verify `N % class_count == 0`.
3. Compute `per_class = N / class_count`.
4. For each class, count the number of `true` bits in its slice.
5. Predicted label = class with the largest count.

This is the plaintext equivalent of the final class vote aggregation used by the exported model.

## Expected Result

The plaintext Rust evaluator should print:

- model summary
- progress through the test set
- final `Accuracy: ...`

If plaintext Rust accuracy does not match the exported model behavior, do not proceed to encrypted evaluation yet. Fix the plaintext path first.

## Recommended Workflow Before Encryption

1. Export the trained model to CSV/JSON artifacts.
2. Run plaintext Rust evaluation on that exported folder.
3. Confirm accuracy is correct.
4. Only then run:
   - `lgn_eval_boolean`
   - `lgn_eval_shortint`
   - `lgn_eval_integer`

This avoids debugging TFHE issues when the export itself is wrong.

## Troubleshooting

- Path errors: make sure the `<run_id>` folder contains all three required files.
- Shape mismatch: confirm `in_dim` matches the number of feature columns in `test_binarized.csv`.
- Wrong model size: check `log_args.txt`, not just folder names.
- Gate parse errors: normalize gate names by lowercasing and removing spaces, underscores, and hyphens.
- Accuracy mismatch: verify the binarization in `best_lgn_metadata.json` matches the dataset used to produce `test_binarized.csv`.
