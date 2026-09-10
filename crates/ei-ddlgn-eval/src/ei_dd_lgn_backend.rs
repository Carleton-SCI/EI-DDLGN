use rayon::prelude::*;
use std::sync::Arc;
use tfhe::boolean::prelude::*;

use crate::model::Gate;

pub type SharedCiphertext = Arc<Ciphertext>;
const PARALLEL_ENCRYPTED_WORK_THRESHOLD: usize = 64;

/// Runtime wire state for encrypted inference with public-constant propagation.
///
/// Inputs begin encrypted. If a gate produces a value known from the gate
/// truth table alone, later layers carry that value as public instead of as a
/// ciphertext.
#[derive(Clone)]
pub enum EiDdlgnWire {
    Public(bool),
    Encrypted(SharedCiphertext),
}

#[derive(Clone, Copy)]
pub struct CompiledEiGate {
    input_a: usize,
    input_b: usize,
    encrypted_eval: EncryptedEval,
    public_public: [bool; 4],
    public_a: [UnaryCollapse; 2],
    public_b: [UnaryCollapse; 2],
}

#[derive(Clone)]
pub struct CompiledEiLayer {
    gates: Vec<CompiledEiGate>,
    has_cheap_encrypted_outputs: bool,
}

pub type CompiledEiLayers = Vec<CompiledEiLayer>;

/// Boolean encrypted-inference backend with public-constant propagation.
///
/// This backend uses TFHE Boolean ciphertexts like `BooleanEngine`, but it
/// stores every runtime wire as either a public constant or an encrypted
/// ciphertext. Mixed public/encrypted gates collapse to 0, 1, x, or !x and
/// therefore skip the two-input encrypted gate operation.
pub struct EiDdlgnEngine {
    pub client_key: ClientKey,
    pub server_key: ServerKey,
}

impl EiDdlgnEngine {
    pub fn new() -> Self {
        let (client_key, server_key) = gen_keys();
        Self {
            client_key,
            server_key,
        }
    }

    pub fn new_with_param(param: BooleanParameters) -> Self {
        let client_key = ClientKey::new(&param);
        let server_key = ServerKey::new(&client_key);
        Self {
            client_key,
            server_key,
        }
    }

    /// Encrypts clear input booleans as encrypted EI_DDLGN wires.
    pub fn encrypt_inputs(&self, input: &[bool]) -> Vec<EiDdlgnWire> {
        input
            .iter()
            .map(|value| EiDdlgnWire::Encrypted(Arc::new(self.client_key.encrypt(*value))))
            .collect()
    }

    /// Converts final EI_DDLGN wires back to booleans.
    ///
    /// Public final wires are returned directly; encrypted final wires are
    /// decrypted with the client key.
    pub fn decrypt_outputs(&self, output: &[EiDdlgnWire]) -> Vec<bool> {
        output
            .iter()
            .map(|wire| match wire {
                EiDdlgnWire::Public(value) => *value,
                EiDdlgnWire::Encrypted(ct) => self.client_key.decrypt(ct.as_ref()),
            })
            .collect()
    }

    /// Precomputes gate truth-table collapse metadata once per model.
    pub fn compile_layers(&self, layers: &[Vec<(usize, usize, Gate)>]) -> CompiledEiLayers {
        layers
            .iter()
            .map(|layer| {
                let gates: Vec<CompiledEiGate> = layer
                    .iter()
                    .map(|(input_a, input_b, gate)| CompiledEiGate::new(*input_a, *input_b, *gate))
                    .collect();
                CompiledEiLayer::new(gates)
            })
            .collect()
    }

    /// Evaluates the network layer by layer with runtime constant propagation.
    pub fn eval_layers(
        &self,
        layers: &[Vec<(usize, usize, Gate)>],
        input: &[EiDdlgnWire],
        use_parallel: bool,
    ) -> Vec<EiDdlgnWire> {
        let compiled = self.compile_layers(layers);
        self.eval_compiled_layers(&compiled, input, use_parallel)
    }

    /// Evaluates a precompiled network layer plan.
    pub fn eval_compiled_layers(
        &self,
        layers: &[CompiledEiLayer],
        input: &[EiDdlgnWire],
        use_parallel: bool,
    ) -> Vec<EiDdlgnWire> {
        let mut x = input.to_vec();

        for layer in layers {
            let out = if let Some(encrypted_x) = collect_encrypted_inputs(&x) {
                self.eval_all_encrypted_layer(layer, &encrypted_x, use_parallel)
            } else if use_parallel {
                self.eval_layer_with_public_inputs_parallel(layer, &x)
            } else {
                self.eval_layer_sequential(layer, &x)
            };
            x = out;
        }

        x
    }

    fn eval_layer_sequential(
        &self,
        layer: &CompiledEiLayer,
        x: &[EiDdlgnWire],
    ) -> Vec<EiDdlgnWire> {
        layer
            .gates
            .iter()
            .map(|gate| self.eval_compiled_gate(gate, &x[gate.input_a], &x[gate.input_b]))
            .collect()
    }

    fn eval_all_encrypted_layer(
        &self,
        layer: &CompiledEiLayer,
        x: &[SharedCiphertext],
        use_parallel: bool,
    ) -> Vec<EiDdlgnWire> {
        if !layer.has_cheap_encrypted_outputs {
            return if use_parallel {
                layer
                    .gates
                    .par_iter()
                    .map(|gate| {
                        self.eval_encrypted_eval(
                            gate.encrypted_eval,
                            &x[gate.input_a],
                            &x[gate.input_b],
                        )
                    })
                    .collect()
            } else {
                layer
                    .gates
                    .iter()
                    .map(|gate| {
                        self.eval_encrypted_eval(
                            gate.encrypted_eval,
                            &x[gate.input_a],
                            &x[gate.input_b],
                        )
                    })
                    .collect()
            };
        }

        let mut out = vec![None; layer.gates.len()];
        let mut work = Vec::new();
        for (output_index, gate) in layer.gates.iter().enumerate() {
            let act = &x[gate.input_a];
            let bct = &x[gate.input_b];
            if let Some(wire) = cheap_encrypted_output(gate.encrypted_eval, act, bct) {
                out[output_index] = Some(wire);
            } else {
                work.push(DeferredEncryptedWork::from_encrypted_eval(
                    output_index,
                    gate.encrypted_eval,
                    act,
                    bct,
                ));
            }
        }

        self.finish_deferred_work(out, work, use_parallel)
    }

    fn eval_layer_with_public_inputs_parallel(
        &self,
        layer: &CompiledEiLayer,
        x: &[EiDdlgnWire],
    ) -> Vec<EiDdlgnWire> {
        let mut out = vec![None; layer.gates.len()];
        let mut work = Vec::new();

        for (output_index, gate) in layer.gates.iter().enumerate() {
            match (&x[gate.input_a], &x[gate.input_b]) {
                (EiDdlgnWire::Public(av), EiDdlgnWire::Public(bv)) => {
                    out[output_index] = Some(EiDdlgnWire::Public(
                        gate.public_public[truth_table_index(*av, *bv)],
                    ));
                }
                (EiDdlgnWire::Public(av), EiDdlgnWire::Encrypted(bct)) => {
                    enqueue_or_set_unary_collapse(
                        &mut out,
                        &mut work,
                        output_index,
                        gate.public_a[usize::from(*av)],
                        bct,
                    );
                }
                (EiDdlgnWire::Encrypted(act), EiDdlgnWire::Public(bv)) => {
                    enqueue_or_set_unary_collapse(
                        &mut out,
                        &mut work,
                        output_index,
                        gate.public_b[usize::from(*bv)],
                        act,
                    );
                }
                (EiDdlgnWire::Encrypted(act), EiDdlgnWire::Encrypted(bct)) => {
                    if let Some(wire) = cheap_encrypted_output(gate.encrypted_eval, act, bct) {
                        out[output_index] = Some(wire);
                    } else {
                        work.push(DeferredEncryptedWork::from_encrypted_eval(
                            output_index,
                            gate.encrypted_eval,
                            act,
                            bct,
                        ));
                    }
                }
            }
        }

        self.finish_deferred_work(out, work, true)
    }

    fn finish_deferred_work(
        &self,
        mut out: Vec<Option<EiDdlgnWire>>,
        work: Vec<DeferredEncryptedWork>,
        use_parallel: bool,
    ) -> Vec<EiDdlgnWire> {
        let evaluated: Vec<(usize, EiDdlgnWire)> =
            if use_parallel && work.len() >= PARALLEL_ENCRYPTED_WORK_THRESHOLD {
                work.par_iter()
                    .map(|job| (job.output_index, self.eval_deferred_work(job)))
                    .collect()
            } else {
                work.iter()
                    .map(|job| (job.output_index, self.eval_deferred_work(job)))
                    .collect()
            };

        for (output_index, wire) in evaluated {
            out[output_index] = Some(wire);
        }

        out.into_iter()
            .map(|wire| wire.expect("every compiled EI gate must produce one output wire"))
            .collect()
    }

    fn eval_compiled_gate(
        &self,
        gate: &CompiledEiGate,
        a: &EiDdlgnWire,
        b: &EiDdlgnWire,
    ) -> EiDdlgnWire {
        match (a, b) {
            (EiDdlgnWire::Public(av), EiDdlgnWire::Public(bv)) => {
                EiDdlgnWire::Public(gate.public_public[truth_table_index(*av, *bv)])
            }
            (EiDdlgnWire::Public(av), EiDdlgnWire::Encrypted(bct)) => {
                self.apply_unary_collapse(gate.public_a[usize::from(*av)], bct)
            }
            (EiDdlgnWire::Encrypted(act), EiDdlgnWire::Public(bv)) => {
                self.apply_unary_collapse(gate.public_b[usize::from(*bv)], act)
            }
            (EiDdlgnWire::Encrypted(act), EiDdlgnWire::Encrypted(bct)) => {
                self.eval_encrypted_eval(gate.encrypted_eval, act, bct)
            }
        }
    }

    fn apply_unary_collapse(
        &self,
        collapse: UnaryCollapse,
        variable: &SharedCiphertext,
    ) -> EiDdlgnWire {
        match collapse {
            UnaryCollapse::Public(value) => EiDdlgnWire::Public(value),
            UnaryCollapse::Identity => EiDdlgnWire::Encrypted(Arc::clone(variable)),
            UnaryCollapse::Not => {
                EiDdlgnWire::Encrypted(Arc::new(self.server_key.not(variable.as_ref())))
            }
        }
    }

    fn eval_encrypted_eval(
        &self,
        eval: EncryptedEval,
        act: &SharedCiphertext,
        bct: &SharedCiphertext,
    ) -> EiDdlgnWire {
        match eval {
            EncryptedEval::Public(value) => EiDdlgnWire::Public(value),
            EncryptedEval::ForwardA => EiDdlgnWire::Encrypted(Arc::clone(act)),
            EncryptedEval::ForwardB => EiDdlgnWire::Encrypted(Arc::clone(bct)),
            EncryptedEval::NotA => {
                EiDdlgnWire::Encrypted(Arc::new(self.server_key.not(act.as_ref())))
            }
            EncryptedEval::NotB => {
                EiDdlgnWire::Encrypted(Arc::new(self.server_key.not(bct.as_ref())))
            }
            EncryptedEval::Binary(op) => {
                EiDdlgnWire::Encrypted(Arc::new(self.eval_binary(op, act, bct)))
            }
        }
    }

    fn eval_deferred_work(&self, job: &DeferredEncryptedWork) -> EiDdlgnWire {
        match &job.work {
            EncryptedWork::Not(ct) => {
                EiDdlgnWire::Encrypted(Arc::new(self.server_key.not(ct.as_ref())))
            }
            EncryptedWork::Binary { op, act, bct } => {
                EiDdlgnWire::Encrypted(Arc::new(self.eval_binary(*op, act, bct)))
            }
        }
    }

    fn eval_binary(
        &self,
        op: EncryptedBinaryOp,
        act: &SharedCiphertext,
        bct: &SharedCiphertext,
    ) -> Ciphertext {
        let act = act.as_ref();
        let bct = bct.as_ref();
        match op {
            EncryptedBinaryOp::And => self.server_key.and(act, bct),
            EncryptedBinaryOp::Or => self.server_key.or(act, bct),
            EncryptedBinaryOp::Xor => self.server_key.xor(act, bct),
            EncryptedBinaryOp::Xnor => self.server_key.xnor(act, bct),
            EncryptedBinaryOp::Nand => self.server_key.nand(act, bct),
            EncryptedBinaryOp::Nor => self.server_key.nor(act, bct),
            EncryptedBinaryOp::Implies => {
                let nota = self.server_key.not(act);
                self.server_key.or(&nota, bct)
            }
            EncryptedBinaryOp::ImpliedBy => {
                let notb = self.server_key.not(bct);
                self.server_key.or(act, &notb)
            }
            EncryptedBinaryOp::NotImplies => {
                let notb = self.server_key.not(bct);
                self.server_key.and(act, &notb)
            }
            EncryptedBinaryOp::NotImpliedBy => {
                let nota = self.server_key.not(act);
                self.server_key.and(&nota, bct)
            }
        }
    }
}

impl CompiledEiLayer {
    fn new(gates: Vec<CompiledEiGate>) -> Self {
        let has_cheap_encrypted_outputs = gates
            .iter()
            .any(|gate| gate.encrypted_eval.is_cheap_output());
        Self {
            gates,
            has_cheap_encrypted_outputs,
        }
    }
}

impl CompiledEiGate {
    fn new(input_a: usize, input_b: usize, gate: Gate) -> Self {
        let public_public = [
            gate_eval_bool(gate, false, false),
            gate_eval_bool(gate, false, true),
            gate_eval_bool(gate, true, false),
            gate_eval_bool(gate, true, true),
        ];
        let public_a = [
            classify_unary(public_public[0], public_public[1]),
            classify_unary(public_public[2], public_public[3]),
        ];
        let public_b = [
            classify_unary(public_public[0], public_public[2]),
            classify_unary(public_public[1], public_public[3]),
        ];

        Self {
            input_a,
            input_b,
            encrypted_eval: EncryptedEval::from_gate(gate),
            public_public,
            public_a,
            public_b,
        }
    }
}

#[derive(Clone)]
struct DeferredEncryptedWork {
    output_index: usize,
    work: EncryptedWork,
}

impl DeferredEncryptedWork {
    fn from_encrypted_eval(
        output_index: usize,
        eval: EncryptedEval,
        act: &SharedCiphertext,
        bct: &SharedCiphertext,
    ) -> Self {
        let work = match eval {
            EncryptedEval::Public(_) | EncryptedEval::ForwardA | EncryptedEval::ForwardB => {
                unreachable!("cheap encrypted outputs should not be deferred")
            }
            EncryptedEval::NotA => EncryptedWork::Not(Arc::clone(act)),
            EncryptedEval::NotB => EncryptedWork::Not(Arc::clone(bct)),
            EncryptedEval::Binary(op) => EncryptedWork::Binary {
                op,
                act: Arc::clone(act),
                bct: Arc::clone(bct),
            },
        };
        Self { output_index, work }
    }
}

#[derive(Clone)]
enum EncryptedWork {
    Not(SharedCiphertext),
    Binary {
        op: EncryptedBinaryOp,
        act: SharedCiphertext,
        bct: SharedCiphertext,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncryptedEval {
    Public(bool),
    ForwardA,
    ForwardB,
    NotA,
    NotB,
    Binary(EncryptedBinaryOp),
}

impl EncryptedEval {
    fn from_gate(gate: Gate) -> Self {
        match gate {
            Gate::Zero => EncryptedEval::Public(false),
            Gate::One => EncryptedEval::Public(true),
            Gate::A => EncryptedEval::ForwardA,
            Gate::B => EncryptedEval::ForwardB,
            Gate::NotA => EncryptedEval::NotA,
            Gate::NotB => EncryptedEval::NotB,
            Gate::And => EncryptedEval::Binary(EncryptedBinaryOp::And),
            Gate::Or => EncryptedEval::Binary(EncryptedBinaryOp::Or),
            Gate::Xor => EncryptedEval::Binary(EncryptedBinaryOp::Xor),
            Gate::NotXor => EncryptedEval::Binary(EncryptedBinaryOp::Xnor),
            Gate::NotAnd => EncryptedEval::Binary(EncryptedBinaryOp::Nand),
            Gate::NotOr => EncryptedEval::Binary(EncryptedBinaryOp::Nor),
            Gate::Implies => EncryptedEval::Binary(EncryptedBinaryOp::Implies),
            Gate::ImpliedBy => EncryptedEval::Binary(EncryptedBinaryOp::ImpliedBy),
            Gate::NotImplies => EncryptedEval::Binary(EncryptedBinaryOp::NotImplies),
            Gate::NotImpliedBy => EncryptedEval::Binary(EncryptedBinaryOp::NotImpliedBy),
        }
    }

    fn is_cheap_output(self) -> bool {
        matches!(
            self,
            EncryptedEval::Public(_) | EncryptedEval::ForwardA | EncryptedEval::ForwardB
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncryptedBinaryOp {
    And,
    Or,
    Xor,
    Xnor,
    Nand,
    Nor,
    Implies,
    ImpliedBy,
    NotImplies,
    NotImpliedBy,
}

fn truth_table_index(a: bool, b: bool) -> usize {
    (usize::from(a) << 1) | usize::from(b)
}

fn gate_eval_bool(gate: Gate, a: bool, b: bool) -> bool {
    match gate {
        Gate::Zero => false,
        Gate::One => true,
        Gate::A => a,
        Gate::B => b,
        Gate::NotA => !a,
        Gate::NotB => !b,
        Gate::And => a & b,
        Gate::Or => a | b,
        Gate::Xor => a ^ b,
        Gate::NotXor => !(a ^ b),
        Gate::NotAnd => !(a & b),
        Gate::NotOr => !(a | b),
        Gate::Implies => !a | b,
        Gate::ImpliedBy => a | !b,
        Gate::NotImplies => a & !b,
        Gate::NotImpliedBy => !a & b,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnaryCollapse {
    Public(bool),
    Identity,
    Not,
}

fn classify_unary(f0: bool, f1: bool) -> UnaryCollapse {
    match (f0, f1) {
        (false, false) => UnaryCollapse::Public(false),
        (true, true) => UnaryCollapse::Public(true),
        (false, true) => UnaryCollapse::Identity,
        (true, false) => UnaryCollapse::Not,
    }
}

fn collect_encrypted_inputs(input: &[EiDdlgnWire]) -> Option<Vec<SharedCiphertext>> {
    let mut encrypted = Vec::with_capacity(input.len());
    for wire in input {
        match wire {
            EiDdlgnWire::Public(_) => return None,
            EiDdlgnWire::Encrypted(ct) => encrypted.push(Arc::clone(ct)),
        }
    }
    Some(encrypted)
}

fn cheap_encrypted_output(
    eval: EncryptedEval,
    act: &SharedCiphertext,
    bct: &SharedCiphertext,
) -> Option<EiDdlgnWire> {
    match eval {
        EncryptedEval::Public(value) => Some(EiDdlgnWire::Public(value)),
        EncryptedEval::ForwardA => Some(EiDdlgnWire::Encrypted(Arc::clone(act))),
        EncryptedEval::ForwardB => Some(EiDdlgnWire::Encrypted(Arc::clone(bct))),
        EncryptedEval::NotA | EncryptedEval::NotB | EncryptedEval::Binary(_) => None,
    }
}

fn enqueue_or_set_unary_collapse(
    out: &mut [Option<EiDdlgnWire>],
    work: &mut Vec<DeferredEncryptedWork>,
    output_index: usize,
    collapse: UnaryCollapse,
    variable: &SharedCiphertext,
) {
    match collapse {
        UnaryCollapse::Public(value) => out[output_index] = Some(EiDdlgnWire::Public(value)),
        UnaryCollapse::Identity => {
            out[output_index] = Some(EiDdlgnWire::Encrypted(Arc::clone(variable)));
        }
        UnaryCollapse::Not => work.push(DeferredEncryptedWork {
            output_index,
            work: EncryptedWork::Not(Arc::clone(variable)),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_GATES: [Gate; 16] = [
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
    ];

    #[test]
    fn mixed_inputs_always_collapse_to_public_identity_or_not() {
        for gate in ALL_GATES {
            for public_value in [false, true] {
                let public_a = classify_unary(
                    gate_eval_bool(gate, public_value, false),
                    gate_eval_bool(gate, public_value, true),
                );
                let public_b = classify_unary(
                    gate_eval_bool(gate, false, public_value),
                    gate_eval_bool(gate, true, public_value),
                );

                assert!(matches!(
                    public_a,
                    UnaryCollapse::Public(_) | UnaryCollapse::Identity | UnaryCollapse::Not
                ));
                assert!(matches!(
                    public_b,
                    UnaryCollapse::Public(_) | UnaryCollapse::Identity | UnaryCollapse::Not
                ));
            }
        }
    }

    #[test]
    fn encrypted_gate_dispatch_is_precompiled() {
        let expected = [
            (Gate::Zero, EncryptedEval::Public(false)),
            (Gate::One, EncryptedEval::Public(true)),
            (Gate::A, EncryptedEval::ForwardA),
            (Gate::B, EncryptedEval::ForwardB),
            (Gate::NotA, EncryptedEval::NotA),
            (Gate::NotB, EncryptedEval::NotB),
            (Gate::And, EncryptedEval::Binary(EncryptedBinaryOp::And)),
            (Gate::Or, EncryptedEval::Binary(EncryptedBinaryOp::Or)),
            (Gate::Xor, EncryptedEval::Binary(EncryptedBinaryOp::Xor)),
            (Gate::NotXor, EncryptedEval::Binary(EncryptedBinaryOp::Xnor)),
            (Gate::NotAnd, EncryptedEval::Binary(EncryptedBinaryOp::Nand)),
            (Gate::NotOr, EncryptedEval::Binary(EncryptedBinaryOp::Nor)),
            (
                Gate::Implies,
                EncryptedEval::Binary(EncryptedBinaryOp::Implies),
            ),
            (
                Gate::ImpliedBy,
                EncryptedEval::Binary(EncryptedBinaryOp::ImpliedBy),
            ),
            (
                Gate::NotImplies,
                EncryptedEval::Binary(EncryptedBinaryOp::NotImplies),
            ),
            (
                Gate::NotImpliedBy,
                EncryptedEval::Binary(EncryptedBinaryOp::NotImpliedBy),
            ),
        ];

        for (gate, encrypted_eval) in expected {
            assert_eq!(EncryptedEval::from_gate(gate), encrypted_eval);
            assert_eq!(
                CompiledEiGate::new(0, 1, gate).encrypted_eval,
                encrypted_eval
            );
        }
    }

    #[test]
    fn compiled_layer_marks_only_public_or_forward_outputs_as_cheap() {
        let fhe_only = CompiledEiLayer::new(vec![
            CompiledEiGate::new(0, 1, Gate::And),
            CompiledEiGate::new(0, 1, Gate::Xor),
            CompiledEiGate::new(0, 1, Gate::NotA),
        ]);
        assert!(!fhe_only.has_cheap_encrypted_outputs);

        let with_forward = CompiledEiLayer::new(vec![
            CompiledEiGate::new(0, 1, Gate::And),
            CompiledEiGate::new(0, 1, Gate::A),
        ]);
        assert!(with_forward.has_cheap_encrypted_outputs);

        let with_public = CompiledEiLayer::new(vec![
            CompiledEiGate::new(0, 1, Gate::Xor),
            CompiledEiGate::new(0, 1, Gate::Zero),
        ]);
        assert!(with_public.has_cheap_encrypted_outputs);
    }
}
