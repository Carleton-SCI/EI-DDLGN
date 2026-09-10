# syntax=docker/dockerfile:1.7
FROM rust:1.93.0-bookworm AS rust-builder
WORKDIR /build
COPY crates/ei-ddlgn-eval crates/ei-ddlgn-eval
RUN cargo build --locked --release \
    --manifest-path crates/ei-ddlgn-eval/Cargo.toml \
    --bin lgn_eval --bin lgn_eval_boolean --bin EI_DDLGN --bin compare_boolean_ei

FROM python:3.11.13-slim-bookworm
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    RAYON_NUM_THREADS=20
WORKDIR /workspace
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt README.md LICENSE NOTICE ./
COPY src src
RUN python -m pip install --no-cache-dir .
COPY artifacts artifacts
COPY paper paper
COPY scripts scripts
COPY --from=rust-builder /build/crates/ei-ddlgn-eval/target/release/lgn_eval /usr/local/bin/lgn_eval
COPY --from=rust-builder /build/crates/ei-ddlgn-eval/target/release/lgn_eval_boolean /usr/local/bin/lgn_eval_boolean
COPY --from=rust-builder /build/crates/ei-ddlgn-eval/target/release/EI_DDLGN /usr/local/bin/EI_DDLGN
COPY --from=rust-builder /build/crates/ei-ddlgn-eval/target/release/compare_boolean_ei /usr/local/bin/compare_boolean_ei
CMD ["python", "scripts/validate_results.py"]
