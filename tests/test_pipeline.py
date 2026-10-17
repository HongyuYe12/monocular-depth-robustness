import json
from pathlib import Path

import numpy as np
from PIL import Image

from depth_wild.evaluate import evaluate_predictions
from depth_wild.io import read_csv, write_csv
from depth_wild.perturb import build_dataset
from depth_wild.plot import plot_metrics


def test_offline_dataset_and_evaluation_pipeline(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pixels = np.tile(np.arange(32, dtype=np.uint8), (24, 1)) * 8
    Image.fromarray(np.repeat(pixels[..., None], 3, axis=2)).save(raw_dir / "scene.png")

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"seed": 7, "perturbations": {"brightness": [0.5], "blur": [], "noise": [], "resolution": []}}),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "source_relpath,scene_type,tags,license\nscene.png,indoor,small-object;boundary,self-captured\n",
        encoding="utf-8",
    )
    manifest_path = build_dataset(raw_dir, dataset_dir, config_path, metadata_path)
    manifest = read_csv(manifest_path)
    assert len(manifest) == 2
    assert {row["scene_type"] for row in manifest} == {"indoor"}
    assert {row["tags"] for row in manifest} == {"small-object;boundary"}
    assert (dataset_dir / "experiment.snapshot.json").is_file()

    prediction_rows = []
    prediction_fields = [
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
    for row in manifest:
        depth = np.tile(np.arange(32, dtype=np.float32), (24, 1))
        depth_path = tmp_path / f"{row['variant_id']}.npy"
        np.save(depth_path, depth)
        prediction_rows.append(
            {
                **{key: row[key] for key in prediction_fields[:10]},
                "model_id": "fake/model",
                "device": "cpu",
                "depth_path": depth_path,
                "visualization_path": "",
                "runtime_seconds": "0.01",
            }
        )
    predictions_path = tmp_path / "predictions.csv"
    write_csv(predictions_path, prediction_rows, prediction_fields)

    metrics_path = evaluate_predictions(predictions_path, tmp_path / "analysis")
    metrics = read_csv(metrics_path)
    assert len(metrics) == 1
    assert float(metrics[0]["spearman"]) > 0.999
    assert float(metrics[0]["affine_nrmse"]) < 1e-6
    assert metrics[0]["scene_type"] == "indoor"

    figures_dir = tmp_path / "figures"
    plot_metrics(metrics_path, figures_dir)
    assert (figures_dir / "summary.csv").is_file()
    assert (figures_dir / "worst_cases.csv").is_file()
    assert (figures_dir / "spearman.png").is_file()
