use rayon::prelude::*;
use tfhe::shortint::parameters::v1_6::V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128;
use tfhe::shortint::server_key::BivariateLookupTableOwned;
use tfhe::shortint::{Ciphertext, ClassicPBSParameters, ClientKey, ServerKey, gen_keys};

use crate::model::Gate;

/// Engine for evaluating Logic Gate Networks using TFHE's Shortint crate.
/// 
/// In this backend, booleans are represented as encrypted 1-bit integers (0 or 1).
/// All logic gates are evaluated using Bivariate Programmable Bootstrapping (PBS)
/// via Lookup Tables (LUTs).
pub struct ShortintEngine {
    pub client_key: ClientKey,
    pub server_key: ServerKey,
    /// Pre-encrypted constant 0.
    ct_zero: Ciphertext,
    /// Pre-encrypted constant 1.
    ct_one: Ciphertext,
    /// Cached Lookup Tables for all 16 binary logic gates.
    gate_luts: Vec<BivariateLookupTableOwned>,
}

impl ShortintEngine {
    /// Creates a new engine with default secure parameters (V1_6, Message 1, Carry 1).
    pub fn new() -> Self {
        let (client_key, server_key) = gen_keys(V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128);
        let ct_zero = client_key.encrypt(0);
        let ct_one = client_key.encrypt(1);
        let gate_luts = build_gate_luts(&server_key);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
            gate_luts,
        }
    }

    /// Creates a new engine with specified TFHE Shortint parameters.
    pub fn new_with_param(param: ClassicPBSParameters) -> Self {
        let (client_key, server_key) = gen_keys(param);
        let ct_zero = client_key.encrypt(0);
        let ct_one = client_key.encrypt(1);
        let gate_luts = build_gate_luts(&server_key);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
            gate_luts,
        }
    }

    /// Encrypts a slice of booleans into a vector of Shortint Ciphertexts.
    /// False maps to encrypt(0), True maps to encrypt(1).
    pub fn encrypt_inputs(&self, input: &[bool]) -> Vec<Ciphertext> {
        input
            .iter()
            .map(|v| self.client_key.encrypt(u64::from(*v)))
            .collect()
    }

    /// Decrypts the output ciphertexts back to booleans.
    /// Checks if the decrypted value is odd (1) to determine True.
    pub fn decrypt_outputs(&self, output: &[Ciphertext]) -> Vec<bool> {
        output
            .iter()
            .map(|ct| {
                let v: u64 = self.client_key.decrypt(ct);
                v % 2 == 1
            })
            .collect()
    }

    /// Evaluates the network layer by layer.
    ///
    /// # Arguments
    /// * `layers` - The network topology.
    /// * `input` - The encrypted input vector.
    /// * `use_parallel` - If true, evaluates gates within a layer in parallel using Rayon.
    pub fn eval_layers(
        &self,
        layers: &[Vec<(usize, usize, Gate)>],
        input: &[Ciphertext],
        use_parallel: bool,
    ) -> Vec<Ciphertext> {
        let mut x = input.to_vec();

        for layer in layers {
            let out: Vec<Ciphertext> = if use_parallel {
                layer
                    .par_iter()
                    .map(|(a, b, g)| self.eval_gate(*g, &x[*a], &x[*b]))
                    .collect()
            } else {
                layer
                    .iter()
                    .map(|(a, b, g)| self.eval_gate(*g, &x[*a], &x[*b]))
                    .collect()
            };
            x = out;
        }

        x
    }

    /// Evaluates a single gate.
    /// Trivial gates (0, 1, Identity) are handled without PBS.
    /// Complex gates use the pre-computed Bivariate LUTs.
    fn eval_gate(&self, gate: Gate, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        match gate {
            Gate::Zero => self.ct_zero.clone(),
            Gate::One => self.ct_one.clone(),
            Gate::A => a.clone(),
            Gate::B => b.clone(),
            _ => {
                let lut = &self.gate_luts[gate_index(gate)];
                self.server_key.apply_lookup_table_bivariate(a, b, lut)
            }
        }
    }
}

const GATE_COUNT: usize = 16;

/// Maps a Gate enum to its index in the `gate_luts` vector.
fn gate_index(gate: Gate) -> usize {
    match gate {
        Gate::Zero => 0,
        Gate::One => 1,
        Gate::A => 2,
        Gate::B => 3,
        Gate::NotA => 4,
        Gate::NotB => 5,
        Gate::And => 6,
        Gate::Or => 7,
        Gate::Xor => 8,
        Gate::NotXor => 9,
        Gate::NotAnd => 10,
        Gate::NotOr => 11,
        Gate::Implies => 12,
        Gate::ImpliedBy => 13,
        Gate::NotImplies => 14,
        Gate::NotImpliedBy => 15,
    }
}

/// Generates Bivariate Lookup Tables for all supported logic gates.
/// This allows evaluating any 2-input boolean function over encrypted integers.
fn build_gate_luts(server_key: &ServerKey) -> Vec<BivariateLookupTableOwned> {
    let mut luts = Vec::with_capacity(GATE_COUNT);
    for gate in [
        Gate::Zero,
        Gate::One,
        Gate::A,
        Gate::B,
        Gate::NotA,
        Gate::NotB,
        Gate::And,
        Gate::Or,
        Gate::Xor,
        Gate::NotXor,
        Gate::NotAnd,
        Gate::NotOr,
        Gate::Implies,
        Gate::ImpliedBy,
        Gate::NotImplies,
        Gate::NotImpliedBy,
    ] {
        let lut = server_key.generate_lookup_table_bivariate(|a, b| gate_eval_u64(gate, a, b));
        luts.push(lut);
    }
    luts
}

/// Helper function defining the truth table for each gate on cleartext integers.
/// Used during LUT generation.
fn gate_eval_u64(gate: Gate, a: u64, b: u64) -> u64 {
    let aa = (a & 1) == 1;
    let bb = (b & 1) == 1;
    let res = match gate {
        Gate::Zero => false,
        Gate::One => true,
        Gate::A => aa,
        Gate::B => bb,
        Gate::NotA => !aa,
        Gate::NotB => !bb,
        Gate::And => aa & bb,
        Gate::Or => aa | bb,
        Gate::Xor => aa ^ bb,
        Gate::NotXor => !(aa ^ bb),
        Gate::NotAnd => !(aa & bb),
        Gate::NotOr => !(aa | bb),
        Gate::Implies => (!aa) | bb,
        Gate::ImpliedBy => aa | (!bb),
        Gate::NotImplies => aa & (!bb),
        Gate::NotImpliedBy => (!aa) & bb,
    };

    u64::from(res)
}
