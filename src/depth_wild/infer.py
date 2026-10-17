from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .io import load_rgb, read_csv, write_csv
from .model import DepthEstimator

PREDICTION_FIELDS = [
    "image_id",
    "source_relpath",
    "scene_type",
    "tags",
    "license",
    "variant_id",
    "perturbation",
    "severity",
    "value",
    "image_path",
    "model_id",
    "device",
    "depth_path",
    "visualization_path",
    "runtime_seconds",
]
MANIFEST_COPY_FIELDS = [
    "image_id",
    "source_relpath",
    "scene_type",
    "tags",
    "license",
    "variant_id",
    "perturbation",
    "severity",
    "value",
    "image_path",
]


def colorize_depth(depth: np.ndarray) -> Image.Image:
    """Create a per-image normalized preview; never use it for metrics."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("depth map has no finite values")
    low, high = np.percentile(finite, [2.0, 98.0])
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0.0, 1.0)
    try:
        from matplotlib import colormaps

        rgb = (colormaps["magma_r"](normalized)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        gray = (normalized * 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
    return Image.fromarray(rgb)


def run_inference(
    manifest_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    device: str,
    limit: int | None = None,
) -> Path:
    manifest_rows = read_csv(manifest_path)
    if limit is not None:
        manifest_rows = manifest_rows[:limit]
    if not manifest_rows:
        raise ValueError("manifest contains no rows")

    raw_dir = output_dir / "raw"
    visualization_dir = output_dir / "visualizations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    existing_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    if predictions_path.is_file():
        for existing in read_csv(predictions_path):
            key = (existing["image_id"], existing["variant_id"], existing["model_id"])
            if Path(existing["depth_path"]).is_file() and Path(existing["visualization_path"]).is_file():
                existing_by_key[key] = existing

    records: list[dict[str, object]] = []
    pending = [
        row
        for row in manifest_rows
        if (row["image_id"], row["variant_id"], model_id) not in existing_by_key
    ]
    estimator = DepthEstimator(model_id=model_id, device=device) if pending else None

    for index, row in enumerate(manifest_rows, start=1):
        key = (row["image_id"], row["variant_id"], model_id)
        if key in existing_by_key:
            records.append(existing_by_key[key])
            write_csv(predictions_path, records, PREDICTION_FIELDS)
            print(f"[{index}/{len(manifest_rows)}] reused {row['image_id']} {row['variant_id']}")
            continue

        image_path = Path(row["image_path"])
        image = load_rgb(image_path)
        started = time.perf_counter()
        assert estimator is not None
        depth = estimator.predict(image)
        elapsed = time.perf_counter() - started

        name = f"{row['image_id']}__{row['variant_id']}"
        depth_path = (raw_dir / f"{name}.npy").resolve()
        visualization_path = (visualization_dir / f"{name}.png").resolve()
        np.save(depth_path, depth, allow_pickle=False)
        colorize_depth(depth).save(visualization_path)
        records.append(
            {
                **{key: row.get(key, "") for key in MANIFEST_COPY_FIELDS},
                "model_id": model_id,
                "device": estimator.device,
                "depth_path": depth_path,
                "visualization_path": visualization_path,
                "runtime_seconds": f"{elapsed:.6f}",
            }
        )
        # Checkpoint after every image because CPU runs can take hours.
        write_csv(predictions_path, records, PREDICTION_FIELDS)
        print(f"[{index}/{len(manifest_rows)}] {row['image_id']} {row['variant_id']} ({elapsed:.2f}s)")

    return predictions_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monocular-depth inference for a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    predictions = run_inference(
        args.manifest,
        args.output,
        model_id=args.model,
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote predictions: {predictions}")


if __name__ == "__main__":
    main()
