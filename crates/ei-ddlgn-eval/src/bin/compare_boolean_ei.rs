use lgn_eval::boolean_backend::BooleanEngine;
use lgn_eval::ei_dd_lgn_backend::EiDdlgnEngine;
use lgn_eval::model::{read_gates, read_meta, Meta};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tfhe::boolean::prelude::*;

const USE_PARALLEL: bool = true;
const DEFAULT_LIMIT: SampleLimit = SampleLimit::Count(5);

#[derive(Clone, Copy, Debug)]
enum SampleLimit {
    Count(usize),
    All,
}

impl SampleLimit {
    fn effective_total(self, total: usize) -> usize {
        match self {
            SampleLimit::Count(limit) => limit.min(total),
            SampleLimit::All => total,
        }
    }

    fn label(self) -> String {
        match self {
            SampleLimit::Count(limit) => limit.to_string(),
            SampleLimit::All => "all".to_string(),
        }
    }
}

#[derive(Debug)]
struct Dataset {
    name: String,
    root: PathBuf,
}

#[derive(Debug)]
struct Args {
    datasets: Vec<Dataset>,
    limit: SampleLimit,
    verbose: bool,
}

#[derive(Clone, Default)]
struct BackendTiming {
    encrypt: Duration,
    eval: Duration,
    decrypt: Duration,
}

impl BackendTiming {
    fn total(&self) -> Duration {
        self.encrypt + self.eval + self.decrypt
    }

    fn avg_eval_seconds(&self, samples: usize) -> f64 {
        if samples == 0 {
            0.0
        } else {
            self.eval.as_secs_f64() / samples as f64
        }
    }

    fn avg_total_seconds(&self, samples: usize) -> f64 {
        if samples == 0 {
            0.0
        } else {
            self.total().as_secs_f64() / samples as f64
        }
    }
}

struct ModelResult {
    depth: usize,
    width: usize,
    samples: usize,
    boolean_correct: usize,
    ei_correct: usize,
    predictions_match: usize,
    boolean_timing: BackendTiming,
    ei_timing: BackendTiming,
}

impl ModelResult {
    fn boolean_accuracy(&self) -> f64 {
        ratio(self.boolean_correct, self.samples)
    }

    fn ei_accuracy(&self) -> f64 {
        ratio(self.ei_correct, self.samples)
    }

    fn prediction_match_rate(&self) -> f64 {
        ratio(self.predictions_match, self.samples)
    }

    fn boolean_avg_eval_seconds(&self) -> f64 {
        self.boolean_timing.avg_eval_seconds(self.samples)
    }

    fn ei_avg_eval_seconds(&self) -> f64 {
        self.ei_timing.avg_eval_seconds(self.samples)
    }

    fn eval_speedup_percent(&self) -> f64 {
        speedup_percent(self.boolean_avg_eval_seconds(), self.ei_avg_eval_seconds())
    }

    fn end_to_end_speedup_percent(&self) -> f64 {
        speedup_percent(
            self.boolean_timing.avg_total_seconds(self.samples),
            self.ei_timing.avg_total_seconds(self.samples),
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;

    println!("Comparing Boolean and EI_DDLGN backends");
    println!("Samples/model: {}", args.limit.label());
    println!("Parallel layer evaluation: {}", USE_PARALLEL);
    println!("Datasets: {}", args.datasets.len());

    println!("Generating Boolean backend keys...");
    let bool_keygen_start = Instant::now();
    let boolean_engine = BooleanEngine::new_with_param(DEFAULT_PARAMETERS_KS_PBS);
    println!(
        "  Boolean keygen: {:.3}s",
        bool_keygen_start.elapsed().as_secs_f64()
    );

    println!("Generating EI_DDLGN backend keys...");
    let ei_keygen_start = Instant::now();
    let ei_engine = EiDdlgnEngine::new_with_param(DEFAULT_PARAMETERS_KS_PBS);
    println!(
        "  EI_DDLGN keygen: {:.3}s",
        ei_keygen_start.elapsed().as_secs_f64()
    );

    for dataset in &args.datasets {
        let rows = evaluate_dataset(&boolean_engine, &ei_engine, dataset, &args)?;
        write_dataset_reports(dataset, &rows)?;
    }

    Ok(())
}

fn evaluate_dataset(
    boolean_engine: &BooleanEngine,
    ei_engine: &EiDdlgnEngine,
    dataset: &Dataset,
    args: &Args,
) -> Result<Vec<ModelResult>, Box<dyn Error>> {
    let model_dirs = discover_model_dirs(&dataset.root)?;
    if model_dirs.is_empty() {
        return Err(format!(
            "No model directories found under {}",
            dataset.root.display()
        )
        .into());
    }

    println!();
    println!("== {} ==", dataset.name);
    println!("root={}", dataset.root.display());
    println!("models={}", model_dirs.len());

    let mut rows = Vec::with_capacity(model_dirs.len());
    for model_dir in model_dirs {
        let result = evaluate_model(boolean_engine, ei_engine, &model_dir, args.limit)?;
        println!(
            "depth={} width={} samples={} bool={:.6}s ei={:.6}s speedup={:+.2}%",
            result.depth,
            result.width,
            result.samples,
            result.boolean_avg_eval_seconds(),
            result.ei_avg_eval_seconds(),
            result.eval_speedup_percent()
        );

        if args.verbose {
            println!(
                "  accuracy bool={:.4} ei={:.4} match={:.4}",
                result.boolean_accuracy(),
                result.ei_accuracy(),
                result.prediction_match_rate()
            );
        }

        rows.push(result);
    }

    Ok(rows)
}

fn evaluate_model(
    boolean_engine: &BooleanEngine,
    ei_engine: &EiDdlgnEngine,
    model_dir: &Path,
    limit: SampleLimit,
) -> Result<ModelResult, Box<dyn Error>> {
    let gates_path = model_dir.join("best_lgn_gates.csv");
    let meta_path = model_dir.join("best_lgn_metadata.json");
    let test_path = model_dir.join("test_binarized.csv");

    let meta = read_meta(
        meta_path
            .to_str()
            .ok_or("metadata path is not valid UTF-8")?,
    )?;
    let layers = read_gates(gates_path.to_str().ok_or("gates path is not valid UTF-8")?)?;
    let compiled_ei_layers = ei_engine.compile_layers(&layers);
    let (depth, width) = depth_width_from_meta_or_path(&meta, model_dir)?;

    let total_test_vectors = csv::Reader::from_path(&test_path)?.records().count();
    let effective_total = limit.effective_total(total_test_vectors);

    let mut rdr = csv::Reader::from_path(test_path)?;
    let mut samples = 0usize;
    let mut boolean_correct = 0usize;
    let mut ei_correct = 0usize;
    let mut predictions_match = 0usize;
    let mut boolean_timing = BackendTiming::default();
    let mut ei_timing = BackendTiming::default();

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

        let enc_start = Instant::now();
        let boolean_input = boolean_engine.encrypt_inputs(&input);
        boolean_timing.encrypt += enc_start.elapsed();

        let eval_start = Instant::now();
        let boolean_out = boolean_engine.eval_layers(&layers, &boolean_input, USE_PARALLEL);
        boolean_timing.eval += eval_start.elapsed();

        let decrypt_start = Instant::now();
        let boolean_clear = boolean_engine.decrypt_outputs(&boolean_out);
        boolean_timing.decrypt += decrypt_start.elapsed();
        let boolean_predicted = decode_output(&boolean_clear, &meta)?;

        let enc_start = Instant::now();
        let ei_input = ei_engine.encrypt_inputs(&input);
        ei_timing.encrypt += enc_start.elapsed();

        let eval_start = Instant::now();
        let ei_out = ei_engine.eval_compiled_layers(&compiled_ei_layers, &ei_input, USE_PARALLEL);
        ei_timing.eval += eval_start.elapsed();

        let decrypt_start = Instant::now();
        let ei_clear = ei_engine.decrypt_outputs(&ei_out);
        ei_timing.decrypt += decrypt_start.elapsed();
        let ei_predicted = decode_output(&ei_clear, &meta)?;

        if boolean_predicted == label {
            boolean_correct += 1;
        }
        if ei_predicted == label {
            ei_correct += 1;
        }
        if boolean_predicted == ei_predicted {
            predictions_match += 1;
        }

        samples += 1;
    }

    Ok(ModelResult {
        depth,
        width,
        samples,
        boolean_correct,
        ei_correct,
        predictions_match,
        boolean_timing,
        ei_timing,
    })
}

fn write_dataset_reports(dataset: &Dataset, rows: &[ModelResult]) -> Result<(), Box<dyn Error>> {
    let boolean_table = format_timing_table(
        "Encrypted Boolean inference time, avg eval seconds/vector",
        rows,
        ModelResult::boolean_avg_eval_seconds,
    )?;
    let ei_table = format_timing_table(
        "EI_DDLGN Boolean inference time, avg eval seconds/vector",
        rows,
        ModelResult::ei_avg_eval_seconds,
    )?;
    let speedup_table = format_timing_table(
        "EI_DDLGN eval speedup over Boolean, percent",
        rows,
        ModelResult::eval_speedup_percent,
    )?;
    let comparison_report = format_comparison_report(dataset, rows)?;

    let boolean_path = dataset.root.join("boolean_sweep_timing.txt");
    let ei_path = dataset.root.join("ei_ddlgn_sweep_timing.txt");
    let speedup_path = dataset.root.join("ei_speedup_percent_timing.txt");
    let comparison_path = dataset.root.join("boolean_vs_ei_comparison.txt");

    fs::write(&boolean_path, &boolean_table)?;
    fs::write(&ei_path, &ei_table)?;
    fs::write(&speedup_path, &speedup_table)?;
    fs::write(&comparison_path, &comparison_report)?;

    println!();
    println!("{}", comparison_report);
    println!("Saved:");
    println!("  {}", boolean_path.display());
    println!("  {}", ei_path.display());
    println!("  {}", speedup_path.display());
    println!("  {}", comparison_path.display());

    Ok(())
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut values = std::env::args().skip(1);
    let mut datasets = Vec::new();
    let mut limit = DEFAULT_LIMIT;
    let mut verbose = false;

    while let Some(arg) = values.next() {
        if arg == "--limit" {
            let value = values.next().ok_or("--limit requires a value")?;
            limit = parse_limit(&value)?;
        } else if let Some(value) = arg.strip_prefix("--limit=") {
            limit = parse_limit(value)?;
        } else if arg == "--dataset" {
            let value = values.next().ok_or("--dataset requires NAME=PATH")?;
            datasets.push(parse_dataset_arg(&value)?);
        } else if let Some(value) = arg.strip_prefix("--dataset=") {
            datasets.push(parse_dataset_arg(value)?);
        } else if arg == "--verbose" {
            verbose = true;
        } else if arg == "--help" || arg == "-h" {
            print_usage();
            std::process::exit(0);
        } else if arg.starts_with('-') {
            return Err(format!("Unknown argument: {}", arg).into());
        } else {
            datasets.push(Dataset {
                name: dataset_name_from_path(&arg),
                root: PathBuf::from(arg),
            });
        }
    }

    if datasets.is_empty() {
        datasets = default_datasets();
    }

    Ok(Args {
        datasets,
        limit,
        verbose,
    })
}

fn print_usage() {
    eprintln!(
        "Usage: compare_boolean_ei [--limit N|all] [--verbose] [--dataset NAME=PATH] [DATASET_ROOT ...]"
    );
    eprintln!(
        "With no dataset arguments, runs MNIST, FashionMNIST, and UCI Phishing depth/width sweeps."
    );
    eprintln!("Default sample limit is 5 per model.");
}

fn parse_limit(value: &str) -> Result<SampleLimit, Box<dyn Error>> {
    if value.eq_ignore_ascii_case("all") {
        Ok(SampleLimit::All)
    } else {
        Ok(SampleLimit::Count(value.parse()?))
    }
}

fn parse_dataset_arg(value: &str) -> Result<Dataset, Box<dyn Error>> {
    let (name, path) = value
        .split_once('=')
        .or_else(|| value.split_once(':'))
        .ok_or("--dataset value must be NAME=PATH")?;
    Ok(Dataset {
        name: name.to_string(),
        root: PathBuf::from(path),
    })
}

fn default_datasets() -> Vec<Dataset> {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let models = base.join("../../artifacts/models");
    vec![
        Dataset {
            name: "MNIST".to_string(),
            root: models.join("mnist"),
        },
        Dataset {
            name: "FashionMNIST".to_string(),
            root: models.join("fashion-mnist"),
        },
        Dataset {
            name: "UCI Phishing Websites".to_string(),
            root: models.join("uci-phishing"),
        },
    ]
}

fn dataset_name_from_path(value: &str) -> String {
    Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dataset")
        .to_string()
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

fn decode_output(out: &[bool], meta: &Meta) -> Result<usize, Box<dyn Error>> {
    if out.len() % meta.class_count != 0 {
        return Err("Last layer size not divisible by class_count".into());
    }

    let per_class = out.len() / meta.class_count;
    let mut best_c = 0usize;
    let mut best_sum = -1isize;

    for c in 0..meta.class_count {
        let mut sum = 0isize;
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

fn format_timing_table(
    title: &str,
    rows: &[ModelResult],
    value_fn: fn(&ModelResult) -> f64,
) -> Result<String, Box<dyn Error>> {
    let mut depths = BTreeSet::new();
    let mut widths = BTreeSet::new();
    let mut by_arch = BTreeMap::new();

    for row in rows {
        depths.insert(row.depth);
        widths.insert(row.width);
        if by_arch
            .insert((row.depth, row.width), value_fn(row))
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
    table.push_str(title);
    table.push('\n');
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
                Some(value) => table.push_str(&format!("{:>14.6}", value)),
                None => table.push_str(&format!("{:>14}", "-")),
            }
        }
        table.push('\n');
    }

    Ok(table)
}

fn format_comparison_report(
    dataset: &Dataset,
    rows: &[ModelResult],
) -> Result<String, Box<dyn Error>> {
    let mut sorted: Vec<&ModelResult> = rows.iter().collect();
    sorted.sort_by_key(|row| (row.depth, row.width));

    let mut out = String::new();
    out.push_str(&format!(
        "Boolean vs EI_DDLGN comparison: {}\n",
        dataset.name
    ));
    out.push_str("Positive speedup means EI_DDLGN is faster than Boolean.\n");
    out.push_str(&format!(
        "{:>5} {:>7} {:>7} {:>10} {:>10} {:>10} {:>12} {:>12} {:>11} {:>11}\n",
        "depth",
        "width",
        "samples",
        "bool_acc",
        "ei_acc",
        "match",
        "bool_eval",
        "ei_eval",
        "eval_%",
        "e2e_%"
    ));

    for row in sorted {
        out.push_str(&format!(
            "{:>5} {:>7} {:>7} {:>10.4} {:>10.4} {:>10.4} {:>12.6} {:>12.6} {:>+10.2}% {:>+10.2}%\n",
            row.depth,
            row.width,
            row.samples,
            row.boolean_accuracy(),
            row.ei_accuracy(),
            row.prediction_match_rate(),
            row.boolean_avg_eval_seconds(),
            row.ei_avg_eval_seconds(),
            row.eval_speedup_percent(),
            row.end_to_end_speedup_percent()
        ));
    }

    let avg_eval_speedup = if rows.is_empty() {
        0.0
    } else {
        rows.iter()
            .map(ModelResult::eval_speedup_percent)
            .sum::<f64>()
            / rows.len() as f64
    };
    out.push_str(&format!(
        "\nMean EI_DDLGN eval speedup across models: {:+.2}%\n",
        avg_eval_speedup
    ));

    Ok(out)
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn speedup_percent(boolean_seconds: f64, ei_seconds: f64) -> f64 {
    if boolean_seconds == 0.0 || ei_seconds == 0.0 {
        0.0
    } else {
        ((boolean_seconds / ei_seconds) - 1.0) * 100.0
    }
}
