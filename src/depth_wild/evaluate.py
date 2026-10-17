from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_csv, write_csv
from .metrics import robustness_metrics

METRIC_FIELDS = [
    "image_id",
    "source_relpath",
    "scene_type",
    "tags",
    "license",
    "variant_id",
    "perturbation",
    "severity",
    "value",
    "model_id",
    "spearman",
    "order_consistency",
    "affine_nrmse",
    "edge_f1",
]
PREDICTION_COPY_FIELDS = [
    "image_id",
    "source_relpath",
    "scene_type",
    "tags",
    "license",
    "variant_id",
    "perturbation",
    "severity",
    "value",
    "model_id",
]


def _row_seed(image_id: str, variant_id: str) -> int:
    digest = hashlib.sha256(f"{image_id}:{variant_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def evaluate_predictions(predictions_path: Path, output_dir: Path) -> Path:
    rows = read_csv(predictions_path)
    if not rows:
        raise ValueError("predictions file contains no rows")
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)

    metric_rows: list[dict[str, object]] = []
    for image_id, group in sorted(by_image.items()):
        originals = [row for row in group if row["perturbation"] == "original"]
        if len(originals) != 1:
            raise ValueError(f"{image_id} must have exactly one original prediction")
        reference = np.load(originals[0]["depth_path"], allow_pickle=False)
        for row in group:
            if row["perturbation"] == "original":
                continue
            prediction = np.load(row["depth_path"], allow_pickle=False)
            metrics = robustness_metrics(
                prediction,
                reference,
                seed=_row_seed(image_id, row["variant_id"]),
            )
            metric_rows.append(
                {
                    **{key: row.get(key, "") for key in PREDICTION_COPY_FIELDS},
                    **{key: f"{value:.8f}" for key, value in metrics.items()},
                }
            )

    if not metric_rows:
        raise ValueError("no perturbed predictions were found")
    metrics_path = output_dir / "metrics.csv"
    write_csv(metrics_path, metric_rows, METRIC_FIELDS)
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare perturbed predictions with original predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(args.predictions, args.output)
    print(f"Wrote metrics: {metrics}")


if __name__ == "__main__":
    main()
