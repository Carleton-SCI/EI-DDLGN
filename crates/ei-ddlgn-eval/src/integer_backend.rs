use rayon::prelude::*;
use tfhe::integer::{gen_keys_radix, RadixCiphertext, RadixClientKey, ServerKey};
use tfhe::shortint::parameters::v1_6::V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128;

use crate::model::Gate;

/// Engine for evaluating Logic Gate Networks using TFHE's Integer crate (Radix decomposition).
///
/// Although designed for integer arithmetic, this backend is used here for boolean logic
/// by treating values as 1-bit integers and using bitwise operations (AND, OR, XOR, NOT).
pub struct IntegerEngine {
    pub client_key: RadixClientKey,
    pub server_key: ServerKey,
    /// Pre-encrypted constant 0 (as RadixCiphertext).
    ct_zero: RadixCiphertext,
    /// Pre-encrypted constant 1 (as RadixCiphertext).
    ct_one: RadixCiphertext,
}

impl IntegerEngine {
    /// Creates a new engine with default parameters.
    /// Uses 1 block per ciphertext since we only need to represent 0 and 1.
    pub fn new() -> Self {
        let (client_key, server_key) = gen_keys_radix(
            V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128,
            1, // num_blocks
        );
        let ct_zero = client_key.encrypt(0u64);
        let ct_one = client_key.encrypt(1u64);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
        }
    }

    /// Creates a new engine with specified parameters.
    pub fn new_with_param(param: tfhe::shortint::parameters::ShortintParameterSet) -> Self {
        let (client_key, server_key) = gen_keys_radix(param, 1);
        let ct_zero = client_key.encrypt(0u64);
        let ct_one = client_key.encrypt(1u64);

        Self {
            client_key,
            server_key,
            ct_zero,
            ct_one,
        }
    }

    /// Encrypts input booleans into RadixCiphertexts.
    pub fn encrypt_inputs(&self, input: &[bool]) -> Vec<RadixCiphertext> {
        input
            .iter()
            .map(|v| self.client_key.encrypt(u64::from(*v)))
            .collect()
    }

    /// Decrypts output RadixCiphertexts to booleans.
    pub fn decrypt_outputs(&self, output: &[RadixCiphertext]) -> Vec<bool> {
        output
            .iter()
            .map(|ct| {
                let v: u64 = self.client_key.decrypt(ct);
                v % 2 == 1
            })
            .collect()
    }

    /// Evaluates the network layer by layer.
    pub fn eval_layers(
        &self,
        layers: &[Vec<(usize, usize, Gate)>],
        input: &[RadixCiphertext],
        use_parallel: bool,
    ) -> Vec<RadixCiphertext> {
        let mut x = input.to_vec();

        for layer in layers {
            let out: Vec<RadixCiphertext> = if use_parallel {
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

    /// Evaluates a single gate using bitwise operations on RadixCiphertexts.
    ///
    /// The integer crate provides bitwise operations like `unchecked_bitand`, `unchecked_bitor`,
    /// `unchecked_bitxor`, and `bitnot`. These are sufficient to implement all 16 logic gates.
    /// 'unchecked' usually implies more efficient operations assuming inputs are valid.
    fn eval_gate(&self, gate: Gate, a: &RadixCiphertext, b: &RadixCiphertext) -> RadixCiphertext {
        match gate {
            Gate::Zero => self.ct_zero.clone(),
            Gate::One => self.ct_one.clone(),
            Gate::A => a.clone(),
            Gate::B => b.clone(),
            Gate::NotA => self.server_key.bitnot(a),
            Gate::NotB => self.server_key.bitnot(b),
            Gate::And => self.server_key.unchecked_bitand(a, b),
            Gate::Or => self.server_key.unchecked_bitor(a, b),
            Gate::Xor => self.server_key.unchecked_bitxor(a, b),
            Gate::NotXor => {
                let tmp = self.server_key.unchecked_bitxor(a, b);
                self.server_key.bitnot(&tmp)
            }
            Gate::NotAnd => {
                let tmp = self.server_key.unchecked_bitand(a, b);
                self.server_key.bitnot(&tmp)
            }
            Gate::NotOr => {
                let tmp = self.server_key.unchecked_bitor(a, b);
                self.server_key.bitnot(&tmp)
            }
            Gate::Implies => {
                // A -> B  ==  !A | B
                let nota = self.server_key.bitnot(a);
                self.server_key.unchecked_bitor(&nota, b)
            }
            Gate::ImpliedBy => {
                // A <- B  ==  A | !B
                let notb = self.server_key.bitnot(b);
                self.server_key.unchecked_bitor(a, &notb)
            }
            Gate::NotImplies => {
                // !(A -> B) == A & !B
                let notb = self.server_key.bitnot(b);
                self.server_key.unchecked_bitand(a, &notb)
            }
            Gate::NotImpliedBy => {
                // !(A <- B) == !A & B
                let nota = self.server_key.bitnot(a);
                self.server_key.unchecked_bitand(&nota, b)
            }
        }
    }
}
