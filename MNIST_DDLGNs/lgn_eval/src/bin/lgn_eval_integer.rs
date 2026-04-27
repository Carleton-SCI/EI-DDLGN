use lgn_eval::integer_backend::IntegerEngine;
use lgn_eval::model::{read_gates, read_meta, Meta};
use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const USE_PARALLEL: bool = true;
const PROGRESS_EVERY: usize = 1;

/// Main entry point for Integer evaluation.
///
/// Steps:
/// 1. Parse CLI args.
/// 2. Load model.
/// 3. Generate TFHE Integer keys (Radix).
/// 4. Evaluate network on encrypted data.
fn main() -> Result<(), Box<dyn Error>> {
    let (base, limit) = parse_args()?;

    let gates_path = base.join("best_lgn_gates.csv");
    let meta_path = base.join("best_lgn_metadata.json");
    let test_path = base.join("test_binarized.csv");

    println!("Step 1/4: Loading model metadata and gates...");
    let meta = read_meta(meta_path.to_str().unwrap())?;
    let layers = read_gates(gates_path.to_str().unwrap())?;
    print_network_summary(&meta, &layers);

    println!("Step 2/4: Generating TFHE Integer keys...");
    let keygen_start = Instant::now();
    let engine = IntegerEngine::new();
    println!("  Key generation done in {:.3}s", keygen_start.elapsed().as_secs_f64());

    println!("Step 3/4: Analyzing test dataset...");
    let total_test_vectors = csv::Reader::from_path(&test_path)?.records().count();
    let effective_total = limit
        .map(|v| v.min(total_test_vectors))
        .unwrap_or(total_test_vectors);
    println!("  Total Test Vectors found: {}", total_test_vectors);
    println!("  Will evaluate: {}", effective_total);

    println!("Step 4/4: Starting encrypted evaluation...");
    let mut rdr = csv::Reader::from_path(test_path)?;

    let mut correct = 0usize;
    let mut total = 0usize;
    let mut total_encrypt = Duration::ZERO;
    let mut total_eval = Duration::ZERO;
    let mut total_decrypt = Duration::ZERO;

    for (idx, result) in rdr.records().enumerate() {
        if idx >= effective_total {
            break;
        }
        let record = result?;
        let label: usize = record[0].parse()?;

        // Input vector (0s and 1s)
        let mut input = vec![false; meta.in_dim];
        for i in 0..meta.in_dim {
            let v: u8 = record[i + 1].parse()?;
            input[i] = v != 0;
        }

        let enc_start = Instant::now();
        let enc_input = engine.encrypt_inputs(&input);
        total_encrypt += enc_start.elapsed();

        let eval_start = Instant::now();
        let out = engine.eval_layers(&layers, &enc_input, USE_PARALLEL);
        total_eval += eval_start.elapsed();

        let decrypt_start = Instant::now();
        let out_clear = engine.decrypt_outputs(&out);
        total_decrypt += decrypt_start.elapsed();
        let predicted = decode_output(&out_clear, &meta)?;

        if predicted == label {
            correct += 1;
        }
        total += 1;

        if total % PROGRESS_EVERY == 0 || total == effective_total {
            let pct = (total as f64 / effective_total as f64) * 100.0;
            let remaining = effective_total - total;
            print!(
                "\rProgress: {:.1}% done, {} test vectors remain   ",
                pct, remaining
            );
            io::stdout().flush().unwrap();
        }
    }

    println!();
    report_stats(correct, total, total_encrypt, total_eval, total_decrypt);

    Ok(())
}

/// Parses command line arguments.
fn parse_args() -> Result<(PathBuf, Option<usize>), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let base = match args.next() {
        Some(value) => PathBuf::from(value),
        None => {
            eprintln!("Usage: lgn_eval_integer <model_dir> [--limit N]");
            std::process::exit(1);
        }
    };

    let mut limit = None;
    while let Some(arg) = args.next() {
        if arg == "--limit" {
            let value = args.next().ok_or("--limit requires a value")?;
            limit = Some(value.parse()?);
        } else if let Some(value) = arg.strip_prefix("--limit=") {
            limit = Some(value.parse()?);
        } else {
            return Err(format!("Unknown argument: {}", arg).into());
        }
    }

    Ok((base, limit))
}

/// Decodes the final layer output via majority vote.
fn decode_output(out: &[bool], meta: &Meta) -> Result<usize, Box<dyn Error>> {
    if out.len() % meta.class_count != 0 {
        return Err("Last layer size not divisible by class_count".into());
    }

    let per_class = out.len() / meta.class_count;
    let mut best_c = 0usize;
    let mut best_sum = -1isize;

    for c in 0..meta.class_count {
        let mut sum = 0isize;
        // Count '1's for class c
        for j in 0..per_class {
            if out[c * per_class + j] {
                sum += 1;
            }
        }
        if sum > best_sum {
            best_sum = sum;
            best_c = c;
        }
    }

    Ok(best_c)
}

fn print_network_summary(meta: &Meta, layers: &[Vec<(usize, usize, lgn_eval::model::Gate)>]) {
    println!("Model Loaded Successfully.");
    println!("========================================");
    println!("Network Summary:");
    println!("  Input Dimension: {}", meta.in_dim);
    println!("  Output Classes:  {}", meta.class_count);
    println!("  Total Layers:    {}", layers.len());
    println!("  Neurons/Layer:   {}", layers[0].len());
    let total_neurons: usize = layers.iter().map(|l| l.len()).sum();
    println!("  Total Neurons:   {}", total_neurons);
    println!("========================================");
}

fn report_stats(
    correct: usize,
    total: usize,
    total_encrypt: Duration,
    total_eval: Duration,
    total_decrypt: Duration,
) {
    if total == 0 {
        println!("No test vectors evaluated.");
        return;
    }
    println!("Evaluation Complete.");
    println!("Accuracy: {:.16}", correct as f64 / total as f64);
    println!("Total encryption time: {:.3}s", total_encrypt.as_secs_f64());
    println!("Total evaluation time: {:.3}s", total_eval.as_secs_f64());
    println!("Total decryption time: {:.3}s", total_decrypt.as_secs_f64());
    if total > 0 {
        let avg_encrypt = total_encrypt.as_secs_f64() / total as f64;
        let avg_eval = total_eval.as_secs_f64() / total as f64;
        let avg_decrypt = total_decrypt.as_secs_f64() / total as f64;
        let avg_end_to_end = avg_encrypt + avg_eval + avg_decrypt;
        println!(
            "Avg encrypt/vector: {:.6}s",
            avg_encrypt
        );
        println!(
            "Avg eval/vector: {:.6}s",
            avg_eval
        );
        println!("Avg decrypt/vector: {:.6}s", avg_decrypt);
        println!("Avg end-to-end/vector: {:.6}s", avg_end_to_end);
    }
}
