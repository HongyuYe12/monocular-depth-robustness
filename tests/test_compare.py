from pathlib import Path

from depth_wild.compare import compare_metrics
from depth_wild.io import read_csv, write_csv


def _metric_row(model_id: str, scene: str, value: str, score: str) -> dict[str, object]:
    return {
        "image_id": "image-1",
        "source_relpath": "image.jpg",
        "scene_type": scene,
        "tags": "boundary",
        "license": "self-captured",
        "variant_id": f"blur-{value}",
        "perturbation": "blur",
        "severity": "1",
        "value": value,
        "model_id": model_id,
        "spearman": score,
        "order_consistency": score,
        "affine_nrmse": str(1.0 - float(score)),
        "edge_f1": score,
    }


def test_compare_metrics_creates_model_and_scene_summaries(tmp_path: Path):
    fields = list(_metric_row("a", "indoor", "1", "0.9").keys())
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_csv(first, [_metric_row("a", "indoor", "1", "0.9")], fields)
    legacy_row = _metric_row("b", "indoor", "1", "0.8")
    for legacy_field in ("scene_type", "tags", "license"):
        legacy_row.pop(legacy_field)
    legacy_fields = [field for field in fields if field not in {"scene_type", "tags", "license"}]
    write_csv(second, [legacy_row], legacy_fields)

    output = tmp_path / "comparison"
    compare_metrics([f"DepthAnything={first}", f"MiDaS={second}"], output)

    summary = read_csv(output / "model_summary.csv")
    scenes = read_csv(output / "scene_summary.csv")
    assert {row["model_label"] for row in summary} == {"DepthAnything", "MiDaS"}
    assert {row["scene_type"] for row in scenes} == {"indoor", "unlabeled"}
    assert (output / "compare_spearman.png").is_file()
