from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_csv, write_csv
from .plot import METRICS


def _parse_input(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("metric input must use LABEL=PATH")
    label, path = specification.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError("metric input must use a non-empty LABEL=PATH")
    return label.strip(), Path(path.strip())


def compare_metrics(specifications: list[str], output_dir: Path, *, include_per_image: bool = False) -> None:
    if len(specifications) < 2:
        raise ValueError("provide at least two LABEL=PATH metric inputs")
    combined: list[dict[str, object]] = []
    for specification in specifications:
        label, path = _parse_input(specification)
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"metrics file is empty: {path}")
        for row in rows:
            combined.append({**row, "model_label": label})

    output_dir.mkdir(parents=True, exist_ok=True)
    # Older experiment files may predate scene metadata columns. Build a
    # stable union so historical and current runs can still be compared.
    field_union: list[str] = []
    for row in combined:
        for key in row:
            if key != "model_label" and key not in field_union:
                field_union.append(key)
    combined_fields = ["model_label", *field_union]
    if include_per_image:
        write_csv(output_dir / "combined_metrics.csv", combined, combined_fields)

    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    scene_grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in combined:
        grouped[(str(row["model_label"]), str(row["perturbation"]), float(row["value"]))].append(row)
        scene = str(row.get("scene_type", "") or "unlabeled")
        scene_grouped[(str(row["model_label"]), scene, str(row["perturbation"]))].append(row)

    summary_fields = ["model_label", "perturbation", "value", "count"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summary_rows: list[dict[str, object]] = []
    for (label, perturbation, value), rows in sorted(grouped.items()):
        record: dict[str, object] = {
            "model_label": label,
            "perturbation": perturbation,
            "value": f"{value:g}",
            "count": len(rows),
        }
        for metric in METRICS:
            samples = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            record[f"{metric}_mean"] = f"{np.nanmean(samples):.8f}"
            record[f"{metric}_std"] = f"{np.nanstd(samples):.8f}"
        summary_rows.append(record)
    write_csv(output_dir / "model_summary.csv", summary_rows, summary_fields)

    scene_fields = ["model_label", "scene_type", "perturbation", "count", *[f"{m}_mean" for m in METRICS]]
    scene_rows: list[dict[str, object]] = []
    for (label, scene, perturbation), rows in sorted(scene_grouped.items()):
        record = {"model_label": label, "scene_type": scene, "perturbation": perturbation, "count": len(rows)}
        for metric in METRICS:
            record[f"{metric}_mean"] = f"{np.nanmean([float(row[metric]) for row in rows]):.8f}"
        scene_rows.append(record)
    write_csv(output_dir / "scene_summary.csv", scene_rows, scene_fields)

    labels = sorted({key[0] for key in grouped})
    perturbations = sorted({key[1] for key in grouped})
    for metric, ylabel in METRICS.items():
        figure, axes = plt.subplots(1, len(perturbations), figsize=(4.4 * len(perturbations), 3.8), squeeze=False)
        for axis, perturbation in zip(axes[0], perturbations):
            for label in labels:
                values = sorted(value for model, kind, value in grouped if model == label and kind == perturbation)
                means = [np.nanmean([float(row[metric]) for row in grouped[(label, perturbation, value)]]) for value in values]
                axis.plot(values, means, marker="o", label=label)
            axis.set_title(perturbation)
            axis.set_xlabel("perturbation value")
            axis.grid(alpha=0.25)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(loc="best")
        figure.tight_layout()
        figure.savefig(output_dir / f"compare_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare robustness metrics from two or more models.")
    parser.add_argument("--metrics", action="append", required=True, help="Repeat LABEL=PATH for each model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-per-image",
        action="store_true",
        help="Also export per-image rows, which may contain source filenames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_metrics(args.metrics, args.output, include_per_image=args.include_per_image)
    print(f"Wrote model comparison: {args.output}")


if __name__ == "__main__":
    main()
