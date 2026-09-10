#!/usr/bin/env python3
"""Regenerate EI-DDLGN result figures and tables from recorded measurements.

This script reproduces the data-driven artifacts in Sections 5 and 7 of the
accepted manuscript (Figures 4--8 and Tables 3 and 5). Conceptual diagrams and
definition tables are maintained in the manuscript source and are not generated
from experiment data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS = ROOT / "artifacts" / "measurements"
DEFAULT_FIGURES = ROOT / "paper" / "figures"
DEFAULT_TABLES = ROOT / "paper" / "tables"

DATASETS = ["MNIST", "FashionMNIST", "UCI Phishing Websites"]
LABELS = {
    "MNIST": "MNIST",
    "FashionMNIST": "FashionMNIST",
    "UCI Phishing Websites": "UCI Phishing",
}
COLORS = {
    "MNIST": "#1f77b4",
    "FashionMNIST": "#ff7f0e",
    "UCI Phishing Websites": "#2ca02c",
}

TITLE_SIZE = 20
LABEL_SIZE = 17
TICK_SIZE = 15
LEGEND_SIZE = 12
SPINE_WIDTH = 1.4
LINE_WIDTH = 2.8
MARKER_SIZE = 7


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": TICK_SIZE,
            "font.weight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=TICK_SIZE, width=SPINE_WIDTH, length=5)
    for tick in axis.get_xticklabels() + axis.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_linewidth(SPINE_WIDTH)


def style_legend(legend: object) -> None:
    if legend is None:
        return
    for item in legend.get_texts():
        item.set_fontsize(LEGEND_SIZE)
        item.set_fontweight("bold")
    if legend.get_title() is not None:
        legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(SPINE_WIDTH)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(figure)


def pareto_frontier(frame: pd.DataFrame, x_column: str, y_column: str) -> pd.DataFrame:
    ordered = frame.sort_values([x_column, y_column], ascending=[True, True])
    selected = []
    best = -np.inf
    for _, row in ordered.iterrows():
        if row[y_column] > best + 1e-12:
            selected.append(row)
            best = row[y_column]
    return pd.DataFrame(selected)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = pd.read_csv(MEASUREMENTS / "pbs_summary.csv")
    selected = pd.read_csv(MEASUREMENTS / "selected_models.csv")
    timings = pd.read_csv(MEASUREMENTS / "encrypted_timings.csv")
    baselines = pd.read_csv(MEASUREMENTS / "qat_fcnn_baselines.csv")

    expected = {(dataset, depth, width) for dataset in DATASETS for depth in range(1, 7) for width in (2000, 4000, 6000, 8000)}
    actual = set(models[["Dataset", "Depth", "Width"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"Expected the 72-model grid; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if set(timings[["dataset_name", "depth", "width"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("Encrypted timing rows do not cover the exact 72-model grid")
    if not (timings["samples"] == 5).all():
        raise ValueError("Recorded paper timings are expected to use five samples per model")
    return models, selected, timings, baselines


def export_selected_table(selected: pd.DataFrame, timings: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = selected.copy()
    frame["Width"] = frame["W"].str.rstrip("K").astype(int) * 1000
    timing_columns = timings.rename(
        columns={"dataset_name": "Dataset", "depth": "D", "width": "Width"}
    )[["Dataset", "D", "Width", "ei_avg_eval_seconds"]]
    frame = frame.merge(timing_columns, on=["Dataset", "D", "Width"], validate="one_to_one")
    sizes = pd.CategoricalDtype(["Small", "Medium", "Large"], ordered=True)
    datasets = pd.CategoricalDtype(DATASETS, ordered=True)
    frame["Model Size"] = frame["Model Size"].astype(sizes)
    frame["Dataset"] = frame["Dataset"].astype(datasets)
    frame = frame.sort_values(["Model Size", "Dataset"])
    frame["Model"] = frame.apply(
        lambda row: f"{row['Model Size']} (${int(row['D'])}\\times {row['W']}$)", axis=1
    )
    frame["Dataset"] = frame["Dataset"].astype(str).map(LABELS)
    table = frame[
        [
            "Model",
            "Dataset",
            "Accuracy (%)",
            "Actual Constant (%)",
            "PBS Before Optimization (%)",
            "PBS After Optimization (%)",
            "PBS Reduction (%)",
            "ei_avg_eval_seconds",
        ]
    ].rename(
        columns={
            "Accuracy (%)": "Acc. (\\%)",
            "Actual Constant (%)": "Public const. (\\%)",
            "PBS Before Optimization (%)": "PBS before (\\%)",
            "PBS After Optimization (%)": "PBS after (\\%)",
            "PBS Reduction (%)": "PBS red. (\\%)",
            "ei_avg_eval_seconds": "EI eval. (s)",
        }
    )
    numeric = table.columns.difference(["Model", "Dataset"])
    table[numeric] = table[numeric].round(2)
    table.to_csv(output_dir / "fps_selected_models_table.csv", index=False)
    latex = table.set_index(["Model", "Dataset"]).to_latex(
        escape=False,
        multirow=True,
        index_names=False,
        float_format="%.2f",
        caption=(
            "Selected EI-DDLGN models across datasets. PBS before/after are the "
            "percentage of model gates requiring bootstrapping before and after "
            "MFW-PBS Bypass. EI evaluation time is measured per sample."
        ),
        label="tab:fps-selected-ei-ddlgn-pbs",
        column_format="llrrrrrr",
        position="t",
    )
    latex = latex.replace("\\begin{table}[t]\n", "\\begin{table}[t]\n\\centering\n")
    (output_dir / "fps_selected_models_table.tex").write_text(latex, encoding="utf-8")
    return table


def figure_pbs_share(models: pd.DataFrame, output_dir: Path, table_dir: Path) -> None:
    source = models.copy()
    source["PBS Gate Share After Reduction (%)"] = 100 * source["Optimized PBS"] / source["Total Gates"]
    source = source[
        ["Model", "Dataset", "Depth", "Width", "Total Gates", "Optimized PBS", "PBS Gate Share After Reduction (%)"]
    ].sort_values(["Dataset", "Depth", "Width"])
    source.to_csv(table_dir / "fps_optimized_pbs_gate_share_heatmap_source.csv", index=False)
    minimum = source["PBS Gate Share After Reduction (%)"].min()
    maximum = source["PBS Gate Share After Reduction (%)"].max()
    color_threshold = minimum + 0.55 * (maximum - minimum)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True, constrained_layout=True)
    image = None
    for axis, dataset in zip(axes, DATASETS):
        group = source[source["Dataset"] == dataset]
        heatmap = group.pivot(index="Depth", columns="Width", values="PBS Gate Share After Reduction (%)")
        image = axis.imshow(heatmap.to_numpy(), cmap="YlGnBu", vmin=minimum, vmax=maximum, aspect="auto")
        axis.set_title(LABELS[dataset], pad=10)
        axis.set_xlabel("Width (K gates/layer)", labelpad=8)
        axis.set_xticks(np.arange(len(heatmap.columns)), [str(int(width / 1000)) for width in heatmap.columns])
        axis.set_yticks(np.arange(len(heatmap.index)), [str(depth) for depth in heatmap.index])
        axis.grid(False)
        style_axis(axis)
        for y, depth in enumerate(heatmap.index):
            for x, width in enumerate(heatmap.columns):
                value = heatmap.loc[depth, width]
                axis.text(x, y, f"{value:.1f}", ha="center", va="center", color="white" if value >= color_threshold else "black", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Depth", labelpad=8)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("Executed PBS share (%)", labelpad=10)
    save_figure(figure, output_dir, "fps_optimized_pbs_gate_share_heatmaps")


def figure_pbs_distribution(models: pd.DataFrame, output_dir: Path, table_dir: Path) -> None:
    values = [models.loc[models["Dataset"] == dataset, "PBS Reduction %"].to_numpy() for dataset in DATASETS]
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    positions = np.arange(1, 4)
    boxes = axis.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showmeans=True)
    for patch, dataset in zip(boxes["boxes"], DATASETS):
        patch.set_facecolor(COLORS[dataset])
        patch.set_alpha(0.35)
        patch.set_edgecolor(COLORS[dataset])
    random = np.random.default_rng(7)
    for position, dataset, dataset_values in zip(positions, DATASETS, values):
        axis.scatter(np.full(len(dataset_values), position) + random.normal(0, 0.045, len(dataset_values)), dataset_values, s=52, color=COLORS[dataset], alpha=0.78)
    axis.set_xticks(positions, [LABELS[dataset] for dataset in DATASETS])
    axis.set_ylabel("PBS bypass rate (%)", labelpad=8)
    style_axis(axis)
    figure.tight_layout()
    save_figure(figure, output_dir, "fps_pbs_reduction_distribution")
    summary = models.groupby("Dataset")["PBS Reduction %"].agg(models="count", mean="mean", median="median", min="min", max="max").loc[DATASETS].round(3)
    summary.to_csv(table_dir / "fps_pbs_reduction_distribution_summary.csv")


def figure_accuracy_pareto(models: pd.DataFrame, output_dir: Path, table_dir: Path) -> None:
    frame = models.assign(**{"Executed PBS (k)": models["Optimized PBS"] / 1000})
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    frontiers = []
    for dataset in DATASETS:
        group = frame[frame["Dataset"] == dataset]
        axis.scatter(group["Executed PBS (k)"], group["Test Accuracy %"], s=70, alpha=0.68, color=COLORS[dataset], label=f"{LABELS[dataset]} models")
        frontier = pareto_frontier(group, "Executed PBS (k)", "Test Accuracy %")
        frontiers.append(frontier.assign(dataset_name=dataset))
        axis.plot(frontier["Executed PBS (k)"], frontier["Test Accuracy %"], marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color=COLORS[dataset], label=f"{LABELS[dataset]} Pareto")
    axis.set_xlabel("Executed PBS count per inference (thousands)", labelpad=8)
    axis.set_ylabel("Test accuracy (%)", labelpad=8)
    style_axis(axis)
    style_legend(axis.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.24)))
    figure.tight_layout()
    save_figure(figure, output_dir, "fps_accuracy_vs_optimized_pbs")
    pd.concat(frontiers, ignore_index=True).to_csv(table_dir / "fps_accuracy_vs_optimized_pbs_pareto_points.csv", index=False)


def figure_accuracy_budget(models: pd.DataFrame, output_dir: Path, table_dir: Path) -> None:
    rows = []
    for dataset in DATASETS:
        running_best = -np.inf
        for _, row in models[models["Dataset"] == dataset].sort_values("Optimized PBS").iterrows():
            running_best = max(running_best, row["Test Accuracy %"])
            rows.append({"Dataset": dataset, "Optimized PBS": row["Optimized PBS"], "Optimized PBS (k)": row["Optimized PBS"] / 1000, "Best accuracy under budget (%)": running_best})
    budget = pd.DataFrame(rows)
    budget.to_csv(table_dir / "fps_best_accuracy_under_optimized_pbs_budget.csv", index=False)
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    for dataset in DATASETS:
        group = budget[budget["Dataset"] == dataset]
        axis.step(group["Optimized PBS (k)"], group["Best accuracy under budget (%)"], where="post", linewidth=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, color=COLORS[dataset], label=LABELS[dataset])
    axis.set_xlabel("Executed PBS budget\n(thousands/vector)", labelpad=8)
    axis.set_ylabel("Best test accuracy (%)", labelpad=8)
    style_axis(axis)
    style_legend(axis.legend(loc="lower right"))
    figure.tight_layout()
    save_figure(figure, output_dir, "fps_best_accuracy_under_optimized_pbs_budget")


def figure_encrypted_time(timings: pd.DataFrame, output_dir: Path, table_dir: Path) -> None:
    source = timings.assign(width_k=timings["width"] / 1000)
    source.to_csv(table_dir / "fps_ei_encrypted_time_vs_width_source.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        group = source[source["dataset_name"] == dataset]
        for depth, depth_group in group.groupby("depth"):
            depth_group = depth_group.sort_values("width")
            axis.plot(depth_group["width_k"], depth_group["ei_avg_eval_seconds"], marker="o", linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=f"D={depth}")
        axis.set_title(LABELS[dataset], pad=10)
        axis.set_xlabel("Width (K gates/layer)", labelpad=8)
        axis.set_xticks([2, 4, 6, 8])
        style_axis(axis)
    axes[0].set_ylabel("EI-DDLGN encrypted eval. time\n(s/vector)", labelpad=8)
    style_legend(axes[-1].legend(title="Depth", loc="center left", bbox_to_anchor=(1.03, 0.5)))
    save_figure(figure, output_dir, "fps_ei_encrypted_time_vs_width")


def export_arithmetic_table(selected: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> None:
    ei = selected[selected["Dataset"] == "MNIST"].copy()
    ei = pd.DataFrame(
        {
            "Model": "EI-DDLGN " + ei["Model"].str.replace(r" \(.*", "", regex=True),
            "Size / active conn.": ei["Model"].str.extract(r"\$(\d+)\\times (\d+)K\$").apply(lambda row: f"{row.iloc[0]} layers, {row.iloc[1]}K width", axis=1),
            "Acc. (\\%)": ei["Acc. (\\%)"],
            "Time/img (s)": ei["EI eval. (s)"],
            "Circuit bit-width": "N/A",
        }
    )
    qat = pd.DataFrame(
        {
            "Model": baselines["model"],
            "Size / active conn.": baselines["active_connections"].astype(str) + " active conn.",
            "Acc. (\\%)": baselines["accuracy_percent"],
            "Time/img (s)": baselines["time_per_image_seconds"],
            "Circuit bit-width": baselines["circuit_bit_width"],
        }
    )
    table = pd.concat([ei, qat], ignore_index=True)
    table.to_csv(output_dir / "fps_arithmetic_comparison_table.csv", index=False)
    latex = table.to_latex(index=False, escape=False, float_format="%.2f", caption="Compact MNIST comparison between EI-DDLGN and reproduced QAT-FCNN arithmetic TFHE baselines.", label="tab:fps-arithmetic-comparison", column_format="llrrl", position="t")
    latex = latex.replace("\\begin{table}[t]\n", "\\begin{table}[t]\n\\centering\n")
    (output_dir / "fps_arithmetic_comparison_table.tex").write_text(latex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    models, selected, timings, baselines = load_inputs()
    selected_table = export_selected_table(selected, timings, args.tables_dir)
    figure_pbs_share(models, args.figures_dir, args.tables_dir)
    figure_pbs_distribution(models, args.figures_dir, args.tables_dir)
    figure_accuracy_pareto(models, args.figures_dir, args.tables_dir)
    figure_accuracy_budget(models, args.figures_dir, args.tables_dir)
    figure_encrypted_time(timings, args.figures_dir, args.tables_dir)
    export_arithmetic_table(selected_table, baselines, args.tables_dir)
    print(f"Reproduced 5 figures (PDF+PNG) in {args.figures_dir}")
    print(f"Reproduced paper result tables in {args.tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
