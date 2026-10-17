from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_csv, write_csv

METRICS = {
    "spearman": "Spearman rank correlation (higher is better)",
    "order_consistency": "Pairwise order consistency (higher is better)",
    "affine_nrmse": "Affine-aligned NRMSE (lower is better)",
    "edge_f1": "Depth-edge F1 (higher is better)",
}


def plot_metrics(metrics_path: Path, output_dir: Path) -> None:
    rows = read_csv(metrics_path)
    if not rows:
        raise ValueError("metrics file contains no rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["perturbation"], float(row["value"]))].append(row)

    perturbations = sorted({key[0] for key in grouped})
    for metric, ylabel in METRICS.items():
        figure, axes = plt.subplots(1, len(perturbations), figsize=(4.2 * len(perturbations), 3.6), squeeze=False)
        for axis, perturbation in zip(axes[0], perturbations):
            values = sorted(value for kind, value in grouped if kind == perturbation)
            means = [np.mean([float(row[metric]) for row in grouped[(perturbation, value)]]) for value in values]
            stds = [np.std([float(row[metric]) for row in grouped[(perturbation, value)]]) for value in values]
            axis.errorbar(values, means, yerr=stds, marker="o", capsize=3)
            axis.set_title(perturbation)
            axis.set_xlabel("perturbation value")
            axis.grid(alpha=0.25)
        axes[0][0].set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(output_dir / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    worst = sorted(rows, key=lambda row: float(row["affine_nrmse"]), reverse=True)[: min(25, len(rows))]
    write_csv(output_dir / "worst_cases.csv", worst, list(rows[0].keys()))

    summary_fields = ["perturbation", "value", "count"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summary_rows: list[dict[str, object]] = []
    for (perturbation, value), group in sorted(grouped.items()):
        summary: dict[str, object] = {
            "perturbation": perturbation,
            "value": f"{value:g}",
            "count": len(group),
        }
        for metric in METRICS:
            samples = np.array([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = f"{np.nanmean(samples):.8f}"
            summary[f"{metric}_std"] = f"{np.nanstd(samples):.8f}"
        summary_rows.append(summary)
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robustness metrics and export summaries.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metrics(args.metrics, args.output)
    print(f"Wrote figures and summaries: {args.output}")


if __name__ == "__main__":
    main()
