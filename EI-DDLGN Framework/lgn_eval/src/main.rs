use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

/// Represents a single row from the `best_lgn_gates.csv` file.
/// This defines a single logic gate in the network.
#[derive(Debug, Deserialize)]
struct GateRow {
    layer: usize,      // The layer index this gate belongs to
    neuron: usize,     // The specific neuron index within the layer
    input_a: usize,    // Index of the first input from the previous layer
    input_b: usize,    // Index of the second input from the previous layer
    gate_name: String, // The type of logic gate (e.g., "and", "xor")
}

/// Metadata structure loaded from `best_lgn_metadata.json`.
#[derive(Debug, Deserialize)]
struct Meta {
    class_count: usize, // Number of output classes (e.g., 10 for MNIST)
    in_dim: usize,      // Dimension of the input vector (e.g., 784 for MNIST)
}

/// Enum representing all supported 2-input boolean logic gates.
#[derive(Clone, Copy)]
enum Gate { 
    Zero, One,         // Constants
    A, B,              // Identity (pass-through)
    NotA, NotB,        // Negation of single inputs
    And, Or, Xor,      // Standard logic
    NotXor,            // XNOR
    NotAnd, NotOr,     // NAND, NOR
    Implies, ImpliedBy, 
    NotImplies, NotImpliedBy 
}

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/// Normalizes gate name strings to handle variations like "NOT_XOR", "notxor", "not-xor".
fn normalize_gate(name: &str) -> String {
    let mut s = name.trim().to_lowercase();
    // Remove common punctuation to standardize the name
    for ch in [" ", "_", "-", "(", ")", ".", ","] { s = s.replace(ch, ""); }
    s
}

/// Parses a string (e.g., "and", "xor") into a `Gate` enum variant.
fn gate_from_name(name: &str) -> Result<Gate, String> {
    let key = match normalize_gate(name).as_str() {
        "zero" | "0" | "false" => "zero",
        "one" | "1" | "true" => "one",
        "a" => "a", "b" => "b",
        "nota" | "not_a" => "not_a",
        "notb" | "not_b" => "not_b",
        "and" => "and", "or" => "or", "xor" => "xor",
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
        "zero" => Gate::Zero, "one" => Gate::One, "a" => Gate::A, "b" => Gate::B,
        "not_a" => Gate::NotA, "not_b" => Gate::NotB, "and" => Gate::And, "or" => Gate::Or,
        "xor" => Gate::Xor, "not_xor" => Gate::NotXor, "not_and" => Gate::NotAnd, "not_or" => Gate::NotOr,
        "implies" => Gate::Implies, "implied_by" => Gate::ImpliedBy,
        "not_implies" => Gate::NotImplies, "not_implied_by" => Gate::NotImpliedBy,
        _ => return Err(format!("Unknown gate name: {}", name)),
    })
}

/// Evaluates a specific logic gate given two boolean inputs.
#[inline(always)]
fn gate_eval(g: Gate, a: bool, b: bool) -> bool {
    match g {
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
        Gate::Implies => (!a) | b, 
        Gate::ImpliedBy => a | (!b),
        Gate::NotImplies => a & (!b), 
        Gate::NotImpliedBy => (!a) & b,
    }
}

/// Reads the gates CSV and reconstructs the network layer by layer.
/// Returns a Vector of Layers, where each Layer is a Vector of (InputA, InputB, GateType).
fn read_gates(path: &str) -> Result<Vec<Vec<(usize, usize, Gate)>>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    
    // Group raw rows by layer index first
    let mut by_layer: BTreeMap<usize, Vec<GateRow>> = BTreeMap::new();
    for res in rdr.deserialize() {
        let row: GateRow = res?;
        by_layer.entry(row.layer).or_default().push(row);
    }
    
    // Construct the final sorted structure
    let mut layers = Vec::new();
    for (_lid, mut rows) in by_layer {
        // Ensure neurons are processed in the correct order (0..N) within the layer
        rows.sort_by_key(|r| r.neuron);
        
        let mut layer = Vec::with_capacity(rows.len());
        for r in rows { 
            layer.push((r.input_a, r.input_b, gate_from_name(&r.gate_name)?)); 
        }
        layers.push(layer);
    }
    Ok(layers)
}

/// Loads the network metadata (input dimensions, class counts).
fn read_meta(path: &str) -> Result<Meta, Box<dyn Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

/// Performs the forward pass (inference) through the network.
/// 
/// # Arguments
/// * `layers` - The network architecture.
/// * `input` - The boolean input vector (e.g., flattened image pixels).
/// 
/// # Returns
/// * `Vec<bool>` - The final output layer's boolean states.
fn eval_layers(layers: &[Vec<(usize, usize, Gate)>], input: &[bool]) -> Vec<bool> {
    // Start with the initial input (Layer 0 input)
    let mut x = input.to_vec();
    
    // Propagate through each layer sequentially
    for layer in layers {
        let mut out = vec![false; layer.len()];
        // Compute each neuron's output in the current layer
        for (j, (a, b, g)) in layer.iter().enumerate() { 
            // Look up inputs in the previous layer's output 'x'
            out[j] = gate_eval(*g, x[*a], x[*b]); 
        }
        // The output of this layer becomes the input for the next
        x = out;
    }
    x
}

// -----------------------------------------------------------------------------
// Main Execution
// -----------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Parse CLI arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: lgn_eval <model_dir>");
        std::process::exit(1);
    }
    let base = PathBuf::from(&args[1]);

    // 2. Define file paths based on the provided model directory
    let gates_path = base.join("best_lgn_gates.csv");
    let meta_path  = base.join("best_lgn_metadata.json");
    let test_path  = base.join("test_binarized.csv");

    // 3. Load model structure and metadata
    println!("Step 1/3: Loading model metadata and gates...");
    let meta = read_meta(meta_path.to_str().unwrap())?;
    let layers = read_gates(gates_path.to_str().unwrap())?;
    
    // Print Network Summary
    println!("Model Loaded Successfully.");
    println!("========================================");
    println!("Network Summary:");
    println!("  Input Dimension: {}", meta.in_dim);
    println!("  Output Classes:  {}", meta.class_count);
    println!("  Total Layers:    {}", layers.len());
    //print the number of neurons/layer (which is fixed in this network)
    println!("  Neurons/Layer:   {}", layers[0].len());
    let total_neurons: usize = layers.iter().map(|l| l.len()).sum();
    println!("  Total Neurons:   {}", total_neurons);
    println!("========================================");

    // 4. Prepare for evaluation
    println!("Step 2/3: Analyzing test dataset...");
    
    // Count total test vectors first for progress bar
    let total_test_vectors = csv::Reader::from_path(&test_path)?.records().count();
    println!("  Total Test Vectors found: {}", total_test_vectors);

    let mut rdr = csv::Reader::from_path(test_path)?;
    let mut correct = 0usize;
    let mut total = 0usize;

    // 5. Iterate through the test dataset
    println!("Step 3/3: Starting evaluation...");
    
    for result in rdr.records() {
        let record = result?;
        
        // Parse Ground Truth Label (first column)
        let label: usize = record[0].parse()?;
        
        // Parse Input Features (subsequent columns)
        let mut input = vec![false; meta.in_dim];
        for i in 0..meta.in_dim {
            let v: u8 = record[i + 1].parse()?;
            input[i] = v != 0; // Binarize: Non-zero is treated as true
        }

        // 6. Run Inference
        let out = eval_layers(&layers, &input);

        // 7. Decode Output (Voting Mechanism)
        // The output layer is divided into chunks, one chunk per class.
        // We sum the 'True' outputs in each chunk. The chunk with the highest sum wins.
        if out.len() % meta.class_count != 0 { 
            return Err("Last layer size not divisible by class_count".into()); 
        }

        let per_class = out.len() / meta.class_count;
        let mut best_c = 0usize;
        let mut best_sum = -1isize;

        for c in 0..meta.class_count {
            let mut sum = 0isize;
            // Sum votes for class 'c'
            for j in 0..per_class { 
                if out[c * per_class + j] { sum += 1; } 
            }
            // Check if this class is the winner
            if sum > best_sum { 
                best_sum = sum; 
                best_c = c; 
            }
        }

        // 8. Update Accuracy Stats
        if best_c == label { correct += 1; }
        total += 1;
        
        // Update Progress Bar
        if total % 50 == 0 || total == total_test_vectors {
            let pct = (total as f64 / total_test_vectors as f64) * 100.0;
            let remaining = total_test_vectors - total;
            print!("\rProgress: {:.1}% done, {} test vectors remain   ", pct, remaining);
            io::stdout().flush().unwrap();
        }
    }
    
    // Clear progress line
    println!();

    // 9. Report Result
    println!("Evaluation Complete.");
    println!("Accuracy: {:.16}", correct as f64 / total as f64);
    Ok(())
}
