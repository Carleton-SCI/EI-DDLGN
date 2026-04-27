use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;

/// Represents a single row in the CSV file defining the network topology.
/// Each row corresponds to one neuron (gate) in a specific layer.
#[derive(Debug, Deserialize)]
struct GateRow {
    /// The layer index (0-based) where this gate resides.
    layer: usize,
    /// The index of this neuron within the layer.
    neuron: usize,
    /// The index of the first input from the previous layer.
    input_a: usize,
    /// The index of the second input from the previous layer.
    input_b: usize,
    /// The string representation of the logic gate (e.g., "AND", "XOR").
    gate_name: String,
}

/// Metadata structure loaded from a JSON file.
/// Contains global network parameters needed for I/O handling.
#[derive(Debug, Deserialize)]
pub struct Meta {
    /// Number of output classes (e.g., 10 for MNIST).
    pub class_count: usize,
    /// Dimension of the input vector (e.g., 784 for 28x28 images).
    pub in_dim: usize,
}

/// Enumeration of all 16 possible binary logic gates.
/// These represent the functions f(a, b) -> bool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gate {
    Zero,         // Always False
    One,          // Always True
    A,            // Pass-through A
    B,            // Pass-through B
    NotA,         // !A
    NotB,         // !B
    And,          // A & B
    Or,           // A | B
    Xor,          // A ^ B
    NotXor,       // !(A ^ B) (XNOR)
    NotAnd,       // !(A & B) (NAND)
    NotOr,        // !(A | B) (NOR)
    Implies,      // !A | B
    ImpliedBy,    // A | !B
    NotImplies,   // A & !B
    NotImpliedBy, // !A & B
}

/// Reads the gate configuration from a CSV file.
///
/// The CSV is expected to have columns: layer, neuron, input_a, input_b, gate_name.
/// Returns a vector of layers, where each layer is a vector of (input_a, input_b, Gate) tuples.
///
/// # Arguments
/// * `path` - Path to the CSV file.
pub fn read_gates(path: &str) -> Result<Vec<Vec<(usize, usize, Gate)>>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut by_layer: BTreeMap<usize, Vec<GateRow>> = BTreeMap::new();

    // Group rows by layer index using a BTreeMap to ensure sorted layer order.
    for res in rdr.deserialize() {
        let row: GateRow = res?;
        by_layer.entry(row.layer).or_default().push(row);
    }

    let mut layers = Vec::new();
    // Iterate through layers in order.
    for (_lid, mut rows) in by_layer {
        // Ensure neurons within a layer are processed in order (0, 1, 2...).
        rows.sort_by_key(|r| r.neuron);
        
        let mut layer = Vec::with_capacity(rows.len());
        for r in rows {
            // Map the string name to the Gate enum and store wiring info.
            layer.push((r.input_a, r.input_b, gate_from_name(&r.gate_name)?));
        }
        layers.push(layer);
    }

    Ok(layers)
}

/// Reads the network metadata from a JSON file.
pub fn read_meta(path: &str) -> Result<Meta, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

/// Parses a string into a `Gate` enum, handling various naming conventions and normalizations.
pub fn gate_from_name(name: &str) -> Result<Gate, String> {
    let key = match normalize_gate(name).as_str() {
        "zero" | "0" | "false" => "zero",
        "one" | "1" | "true" => "one",
        "a" => "a",
        "b" => "b",
        "nota" | "not_a" => "not_a",
        "notb" | "not_b" => "not_b",
        "and" => "and",
        "or" => "or",
        "xor" => "xor",
        "notxor" | "not_xor" => "not_xor",
        "implies" => "implies",
        "impliedby" | "implied_by" => "implied_by",
        "notand" | "not_and" => "not_and",
        "notor" | "not_or" => "not_or",
        "notimplies" | "not_implies" => "not_implies",
        "notimpliedby" | "not_implied_by" => "not_implied_by",
        _ => return Err(format!("Unknown gate name: {}", name)),
    };

    Ok(match key {
        "zero" => Gate::Zero,
        "one" => Gate::One,
        "a" => Gate::A,
        "b" => Gate::B,
        "not_a" => Gate::NotA,
        "not_b" => Gate::NotB,
        "and" => Gate::And,
        "or" => Gate::Or,
        "xor" => Gate::Xor,
        "not_xor" => Gate::NotXor,
        "not_and" => Gate::NotAnd,
        "not_or" => Gate::NotOr,
        "implies" => Gate::Implies,
        "implied_by" => Gate::ImpliedBy,
        "not_implies" => Gate::NotImplies,
        "not_implied_by" => Gate::NotImpliedBy,
        _ => return Err(format!("Unknown gate name: {}", name)),
    })
}

/// Normalizes a gate name string by lowercasing and removing punctuation/whitespace.
/// This allows matching "NOT_AND", "not-and", "NAND", etc. (if they mapped to supported keys).
fn normalize_gate(name: &str) -> String {
    let mut s = name.trim().to_lowercase();
    for ch in [" ", "_", "-", "(", ")", ".", ","] {
        s = s.replace(ch, "");
    }
    s
}
