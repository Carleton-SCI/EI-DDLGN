//! # LGN Eval
//!
//! This library provides the backend engines for evaluating Logic Gate Networks (LGNs)
//! using Fully Homomorphic Encryption (FHE) via the `tfhe-rs` crate.
//!
//! It supports TFHE backends for encrypted logic-gate evaluation:
//! - **Boolean**: Uses native boolean gates (`tfhe::boolean`).
//! - **EI_DDLGN**: Uses native boolean gates with runtime public-constant propagation.
//! - **Shortint**: Uses 2-bit integers and Programmable Bootstrapping (`tfhe::shortint`).
//! - **Integer**: Uses Radix decomposition (though effectively 1-bit) and bitwise ops (`tfhe::integer`).
//!
//! Each backend implements the same logic gate evaluation interface defined by the model.

pub mod boolean_backend;
pub mod ei_dd_lgn_backend;
pub mod integer_backend;
pub mod model;
pub mod shortint_backend;
