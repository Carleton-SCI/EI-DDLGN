# Contributing

Please keep changes reproducible and scoped to the EI-DDLGN paper code.

1. Create a focused branch.
2. Run `python -m pytest`, `python scripts/validate_results.py`, and `cargo test --locked --lib --manifest-path crates/ei-ddlgn-eval/Cargo.toml`.
3. Regenerate paper outputs when measurement or plotting code changes.
4. Describe whether a result is deterministic (counts/predictions) or hardware-dependent (timing).
5. Do not commit private data, credentials, build directories, or large duplicate datasets.

Changes to recorded measurements should include the command, hardware, thread count, sample count, and raw report used to produce them.
