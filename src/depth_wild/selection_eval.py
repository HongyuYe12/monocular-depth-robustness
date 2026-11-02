from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .focus import rank_subjects
from .subject import SubjectCandidate

DEPTH_COMPONENTS = {"depth_separation", "depth_consistency"}

# 该工具用于回答项目最关键的实验问题：加入深度信息后，主体选择正确率
# 是否优于只使用类别、检测置信度和构图信息的基线。


def _load_scene(path: Path) -> tuple[np.ndarray, list[SubjectCandidate], str]:
    """读取一个人工标注场景及其预先保存的深度图。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    depth_path = Path(payload["depth_path"])
    if not depth_path.is_absolute():
        depth_path = path.parent / depth_path
    depth = np.load(depth_path, allow_pickle=False)
    subjects = [
        SubjectCandidate(
            subject_id=str(row["subject_id"]),
            label=str(row["label"]).lower(),
            confidence=float(row["confidence"]),
            box=tuple(int(value) for value in row["box"]),
        )
        for row in payload["subjects"]
    ]
    return depth, subjects, str(payload["target_subject_id"])


def evaluate_selection(
    scenes: list[Path],
    *,
    preferences: dict[str, float],
    weights: dict[str, float],
    ambiguity_threshold: float,
    larger_depth_is_near: bool,
) -> dict[str, object]:
    """对同一批场景运行无深度与有深度两种主体排序。"""
    # 消融基线只把深度相关权重清零，其他条件完全一致，确保比较公平。
    no_depth_weights = {key: (0.0 if key in DEPTH_COMPONENTS else value) for key, value in weights.items()}
    rows: list[dict[str, object]] = []

    for scene_path in scenes:
        depth, subjects, target_id = _load_scene(scene_path)
        # A 方案：检测置信度、类别偏好、中心位置和面积。
        baseline = rank_subjects(
            depth,
            subjects,
            preferences=preferences,
            weights=no_depth_weights,
            ambiguity_threshold=ambiguity_threshold,
            larger_depth_is_near=larger_depth_is_near,
        )
        # B 方案：在 A 方案基础上加入主体背景分离度和深度一致性。
        depth_guided = rank_subjects(
            depth,
            subjects,
            preferences=preferences,
            weights=weights,
            ambiguity_threshold=ambiguity_threshold,
            larger_depth_is_near=larger_depth_is_near,
        )
        baseline_id = baseline[0].subject.subject_id if baseline else None
        depth_id = depth_guided[0].subject.subject_id if depth_guided else None
        rows.append(
            {
                "scene": str(scene_path),
                "target_subject_id": target_id,
                "baseline_selected": baseline_id,
                "depth_selected": depth_id,
                "baseline_correct": baseline_id == target_id,
                "depth_correct": depth_id == target_id,
                "selection_changed": baseline_id != depth_id,
            }
        )

    total = len(rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    depth_correct = sum(bool(row["depth_correct"]) for row in rows)
    # accuracy_delta > 0 才能支持“深度模块改善主体选择”的实验结论。
    return {
        "scene_count": total,
        "baseline_accuracy": baseline_correct / total if total else None,
        "depth_guided_accuracy": depth_correct / total if total else None,
        "accuracy_delta": (depth_correct - baseline_correct) / total if total else None,
        "changed_selection_count": sum(bool(row["selection_changed"]) for row in rows),
        "scenes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare subject selection with and without depth features.")
    parser.add_argument("--scenes", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/autofocus.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = evaluate_selection(
        args.scenes,
        preferences={key.lower(): float(value) for key, value in config["subject_preferences"].items()},
        weights={key: float(value) for key, value in config["score_weights"].items()},
        ambiguity_threshold=float(config.get("ambiguity_threshold", 0.22)),
        larger_depth_is_near=bool(config.get("larger_depth_is_near", True)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote selection ablation: {args.output}")


if __name__ == "__main__":
    main()
