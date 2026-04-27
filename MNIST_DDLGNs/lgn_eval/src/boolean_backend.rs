use rayon::prelude::*;
use tfhe::boolean::{prelude::*};

use crate::model::Gate;

/// Engine for evaluating Logic Gate Networks using TFHE's Boolean crate.
///
/// This is the most direct mapping as the crate natively supports encrypted boolean gates.
pub struct BooleanEngine {
    pub client_key: ClientKey,
    pub server_key: ServerKey,
    /// Pre-encrypted constant False (trivial encryption).
    ct_zero: Ciphertext,
    /// Pre-encrypted constant True (trivial encryption).
    ct_one: Ciphertext,
}

impl BooleanEngine {
    /// Creates a new engine with default parameters.
    /// Trivial encryption (no noise) is used for constants 0 and 1.
    pub fn new() -> Self {
        let (client_key, server_key) = gen_keys();
        let ct_zero = server_key.trivial_encrypt(false);
        let ct_one = server_key.trivial_encrypt(true);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
        }
    }

    /// Creates a new engine with specified boolean parameters.
    pub fn new_with_param(param: BooleanParameters) -> Self {
        let client_key = ClientKey::new(&param);
        let server_key = ServerKey::new(&client_key);
        let ct_zero = server_key.trivial_encrypt(false);
        let ct_one = server_key.trivial_encrypt(true);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
        }
    }

    /// Encrypts input booleans into boolean Ciphertexts.
    pub fn encrypt_inputs(&self, input: &[bool]) -> Vec<Ciphertext> {
        input.iter().map(|v| self.client_key.encrypt(*v)).collect()
    }

    /// Decrypts boolean Ciphertexts.
    pub fn decrypt_outputs(&self, output: &[Ciphertext]) -> Vec<bool> {
        output.iter().map(|ct| self.client_key.decrypt(ct)).collect()
    }

    /// Evaluates the network layer by layer.
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

    /// Evaluates a single gate using native boolean FHE operations.
    ///
    /// The boolean crate supports `not`, `and`, `or`, `xor`, `xnor`, `nand`, `nor`.
    /// Other gates are composed from these primitives.
    fn eval_gate(&self, gate: Gate, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        match gate {
            Gate::Zero => self.ct_zero.clone(),
            Gate::One => self.ct_one.clone(),
            Gate::A => a.clone(),
            Gate::B => b.clone(),
            Gate::NotA => self.server_key.not(a),
            Gate::NotB => self.server_key.not(b),
            Gate::And => self.server_key.and(a, b),
            Gate::Or => self.server_key.or(a, b),
            Gate::Xor => self.server_key.xor(a, b),
            Gate::NotXor => self.server_key.xnor(a, b),
            Gate::NotAnd => self.server_key.nand(a, b),
            Gate::NotOr => self.server_key.nor(a, b),
            Gate::Implies => {
                // A -> B == !A | B
                let nota = self.server_key.not(a);
                self.server_key.or(&nota, b)
            }
            Gate::ImpliedBy => {
                // A <- B == A | !B
                let notb = self.server_key.not(b);
                self.server_key.or(a, &notb)
            }
            Gate::NotImplies => {
                // !(A -> B) == A & !B
                let notb = self.server_key.not(b);
                self.server_key.and(a, &notb)
            }
            Gate::NotImpliedBy => {
                // !(A <- B) == !A & B
                let nota = self.server_key.not(a);
                self.server_key.and(&nota, b)
            }
        }
    }
}
