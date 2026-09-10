.PHONY: test validate figures rust-test check

test:
	python -m pytest

validate:
	python scripts/validate_results.py

figures:
	python scripts/reproduce_paper.py

rust-test:
	cargo test --locked --lib --manifest-path crates/ei-ddlgn-eval/Cargo.toml

check: test validate figures rust-test
