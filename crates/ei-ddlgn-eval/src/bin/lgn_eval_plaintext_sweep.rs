use lgn_eval::model::{read_gates, read_meta, Gate, Meta};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Args {
    root: PathBuf,
    limit: Option<usize>,
    output: Option<PathBuf>,
    verbose: bool,
}

#[derive(Debug)]
struct ModelTiming {
    depth: usize,
    width: usize,
    samples: usize,
    correct: usize,
    total_eval: Duration,
}

impl ModelTiming {
    fn accuracy(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.correct as f64 / self.samples as f64
        }
    }

    fn avg_eval_seconds(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.total_eval.as_secs_f64() / self.samples as f64
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let model_dirs = discover_model_dirs(&args.root)?;
    if model_dirs.is_empty() {
        return Err(format!(
            "No model directories found under {}. Expected folders containing best_lgn_gates.csv, best_lgn_metadata.json, and test_binarized.csv.",
            args.root.display()
        )
        .into());
    }

    if args.verbose {
        eprintln!("Found {} model directories.", model_dirs.len());
    }

    let mut rows = Vec::with_capacity(model_dirs.len());
    for model_dir in model_dirs {
        rows.push(evaluate_model(&model_dir, args.limit, args.verbose)?);
    }

    let table = format_timing_table(&rows)?;
    print!("{table}");

    let output_path = args
        .output
        .unwrap_or_else(|| args.root.join("plaintext_sweep_timing.txt"));
    fs::write(&output_path, &table)?;
    println!("Saved timing table to: {}", output_path.display());

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut values = std::env::args().skip(1);
    let root = match values.next() {
        Some(value) => PathBuf::from(value),
        None => {
            eprintln!(
                "Usage: lgn_eval_plaintext_sweep <MODEL_SWEEP_DIR> [--limit N] [--output PATH] [--verbose]"
            );
            std::process::exit(1);
        }
    };

    let mut limit = None;
    let mut output = None;
    let mut verbose = false;
    while let Some(arg) = values.next() {
        if arg == "--limit" {
            let value = values.next().ok_or("--limit requires a value")?;
            limit = Some(value.parse()?);
        } else if let Some(value) = arg.strip_prefix("--limit=") {
            limit = Some(value.parse()?);
        } else if arg == "--output" {
            let value = values.next().ok_or("--output requires a value")?;
            output = Some(PathBuf::from(value));
        } else if let Some(value) = arg.strip_prefix("--output=") {
            output = Some(PathBuf::from(value));
        } else if arg == "--verbose" {
            verbose = true;
        } else {
            return Err(format!("Unknown argument: {}", arg).into());
        }
    }

    Ok(Args {
        root,
        limit,
        output,
        verbose,
    })
}

fn discover_model_dirs(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut out = Vec::new();
    collect_model_dirs(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_model_dirs(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn Error>> {
    if is_model_dir(path) {
        out.push(path.to_path_buf());
        return Ok(());
    }

    if !path.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let child = entry?.path();
        if child.is_dir() {
            collect_model_dirs(&child, out)?;
        }
    }

    Ok(())
}

fn is_model_dir(path: &Path) -> bool {
    path.join("best_lgn_gates.csv").is_file()
        && path.join("best_lgn_metadata.json").is_file()
        && path.join("test_binarized.csv").is_file()
}

fn evaluate_model(
    model_dir: &Path,
    limit: Option<usize>,
    verbose: bool,
) -> Result<ModelTiming, Box<dyn Error>> {
    let gates_path = model_dir.join("best_lgn_gates.csv");
    let meta_path = model_dir.join("best_lgn_metadata.json");
    let test_path = model_dir.join("test_binarized.csv");

    let meta = read_meta(
        meta_path
            .to_str()
            .ok_or("metadata path is not valid UTF-8")?,
    )?;
    let layers = read_gates(gates_path.to_str().ok_or("gates path is not valid UTF-8")?)?;
    let (depth, width) = depth_width_from_meta_or_path(&meta, model_dir)?;

    let total_test_vectors = csv::Reader::from_path(&test_path)?.records().count();
    let effective_total = limit
        .map(|value| value.min(total_test_vectors))
        .unwrap_or(total_test_vectors);

    if verbose {
        eprintln!(
            "Evaluating depth={} width={} samples={} dir={}",
            depth,
            width,
            effective_total,
            model_dir.display()
        );
    }

    let mut rdr = csv::Reader::from_path(test_path)?;
    let mut samples = 0usize;
    let mut correct = 0usize;
    let mut total_eval = Duration::ZERO;

    for (idx, result) in rdr.records().enumerate() {
        if idx >= effective_total {
            break;
        }
        let record = result?;
        let label: usize = record[0].parse()?;

        let mut input = vec![false; meta.in_dim];
        for i in 0..meta.in_dim {
            let value: u8 = record[i + 1].parse()?;
            input[i] = value != 0;
        }

        let eval_start = Instant::now();
        let out = eval_layers(&layers, &input);
        total_eval += eval_start.elapsed();

        let predicted = decode_votes(&out, meta.class_count)?;
        if predicted == label {
            correct += 1;
        }
        samples += 1;
    }

    let timing = ModelTiming {
        depth,
        width,
        samples,
        correct,
        total_eval,
    };

    if verbose {
        eprintln!(
            "Finished depth={} width={} accuracy={:.4}% avg_eval={:.9}s/vector",
            depth,
            width,
            timing.accuracy() * 100.0,
            timing.avg_eval_seconds()
        );
    }

    Ok(timing)
}

fn eval_layers(layers: &[Vec<(usize, usize, Gate)>], input: &[bool]) -> Vec<bool> {
    let mut x = input.to_vec();
    for layer in layers {
        let mut out = vec![false; layer.len()];
        for (j, (a, b, gate)) in layer.iter().enumerate() {
            out[j] = gate_eval(*gate, x[*a], x[*b]);
        }
        x = out;
    }
    x
}

#[inline(always)]
fn gate_eval(gate: Gate, a: bool, b: bool) -> bool {
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
        Gate::Implies => (!a) | b,
        Gate::ImpliedBy => a | (!b),
        Gate::NotImplies => a & (!b),
        Gate::NotImpliedBy => (!a) & b,
    }
}

fn decode_votes(out: &[bool], class_count: usize) -> Result<usize, Box<dyn Error>> {
    if class_count == 0 || out.len() % class_count != 0 {
        return Err("Last layer size not divisible by class_count".into());
    }

    let per_class = out.len() / class_count;
    let mut best_class = 0usize;
    let mut best_sum = 0usize;

    for class_idx in 0..class_count {
        let start = class_idx * per_class;
        let sum = out[start..start + per_class]
            .iter()
            .filter(|value| **value)
            .count();
        if class_idx == 0 || sum > best_sum {
            best_sum = sum;
            best_class = class_idx;
        }
    }

    Ok(best_class)
}

fn depth_width_from_meta_or_path(
    meta: &Meta,
    model_dir: &Path,
) -> Result<(usize, usize), Box<dyn Error>> {
    match (meta.num_layers, meta.num_neurons) {
        (Some(depth), Some(width)) => Ok((depth, width)),
        _ => depth_width_from_path(model_dir),
    }
}

fn depth_width_from_path(model_dir: &Path) -> Result<(usize, usize), Box<dyn Error>> {
    let name = model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("model directory name is not valid UTF-8")?;

    let depth = parse_number_after(name, "depth_")
        .ok_or_else(|| format!("Could not parse depth from directory name: {}", name))?;
    let width = parse_number_after(name, "width_")
        .ok_or_else(|| format!("Could not parse width from directory name: {}", name))?;

    Ok((depth, width))
}

fn parse_number_after(value: &str, marker: &str) -> Option<usize> {
    let start = value.find(marker)? + marker.len();
    let digits: String = value[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn format_timing_table(rows: &[ModelTiming]) -> Result<String, Box<dyn Error>> {
    let mut depths = BTreeSet::new();
    let mut widths = BTreeSet::new();
    let mut by_arch = BTreeMap::new();

    for row in rows {
        depths.insert(row.depth);
        widths.insert(row.width);
        if by_arch
            .insert((row.depth, row.width), row.avg_eval_seconds())
            .is_some()
        {
            return Err(format!(
                "Duplicate timing result for depth={} width={}",
                row.depth, row.width
            )
            .into());
        }
    }

    let mut table = String::new();
    table.push_str("Plaintext inference time, avg eval seconds/vector\n");
    table.push_str(&format!("{:>8}", "width"));
    for width in &widths {
        table.push_str(&format!("{:>14}", width));
    }
    table.push('\n');
    table.push_str(&format!("{:>8}\n", "depth"));
    for depth in &depths {
        table.push_str(&format!("{:>8}", depth));
        for width in &widths {
            match by_arch.get(&(*depth, *width)) {
                Some(value) => table.push_str(&format!("{:>14.9}", value)),
                None => table.push_str(&format!("{:>14}", "-")),
            }
        }
        table.push('\n');
    }

    Ok(table)
}
