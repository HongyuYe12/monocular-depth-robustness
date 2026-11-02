from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera_input import load_camera_frame
from .focus import RankedSubject, rank_subjects
from .infer import colorize_depth
from .model import DepthEstimator
from .subject import SubjectDetector

# 本文件负责编排深度引导自动对焦的第一阶段流程：输入帧 → 主体识别 → 深度估计
# → 主体排序 → 初始焦距区域与是否需要焦点扫描。


def _load_config(path: Path) -> dict[str, object]:
    """读取用户类别偏好、评分权重和深度阈值。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _draw_overlay(image: Image.Image, ranked: list[RankedSubject]) -> Image.Image:
    """在原图上画出主体排名、初始焦距区域和歧义提示。"""
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    colors = ["#00E5FF", "#FFD166", "#EF476F", "#06D6A0"]
    for index, item in enumerate(ranked):
        color = colors[index % len(colors)]
        # 排名第一的主体使用更粗的框，表示当前系统选择的对焦目标。
        draw.rectangle(item.subject.box, outline=color, width=4 if index == 0 else 2)
        ambiguity = " ambiguous" if item.depth.multi_depth_ambiguity else ""
        text = (
            f"#{index + 1} {item.subject.label} score={item.total_score:.2f} "
            f"focus={item.focus_zone}{ambiguity}"
        )
        x1, y1, _, _ = item.subject.box
        text_box = draw.textbbox((x1, max(0, y1 - 18)), text, font=ImageFont.load_default())
        draw.rectangle(text_box, fill="#111111")
        draw.text((x1, max(0, y1 - 18)), text, fill=color, font=ImageFont.load_default())
    return overlay


def analyze_frame(
    image_path: Path,
    output_dir: Path,
    *,
    config_path: Path,
    depth_model: str,
    detector_model: str,
    device: str,
) -> Path:
    """分析一张临时相机帧，并写出机器可读与可视化结果。"""
    config = _load_config(config_path)
    # RAW 只在输入层显影；检测与深度模型仍接收统一的 RGB 图像。
    image = load_camera_frame(image_path)
    detector = SubjectDetector(
        model_id=detector_model,
        device=device,
        threshold=float(config.get("minimum_detection_score", 0.55)),
        nms_iou_threshold=float(config.get("detection_nms_iou_threshold", 0.55)),
    )
    depth_estimator = DepthEstimator(model_id=depth_model, device=device)

    # 主体检测和深度估计彼此独立，便于后续更换其中任意一个模型。
    subjects = detector.detect(image)
    depth = depth_estimator.predict(image)
    ranked = rank_subjects(
        depth,
        subjects,
        preferences={key.lower(): float(value) for key, value in dict(config["subject_preferences"]).items()},
        weights={key: float(value) for key, value in dict(config["score_weights"]).items()},
        ambiguity_threshold=float(config.get("ambiguity_threshold", 0.22)),
        larger_depth_is_near=bool(config.get("larger_depth_is_near", True)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "depth.npy", depth, allow_pickle=False)
    colorize_depth(depth).save(output_dir / "depth.png")
    _draw_overlay(image, ranked).save(output_dir / "subjects.png")

    selected = ranked[0] if ranked else None
    # 以下三种情况不应直接采信初始焦距建议：没有主体、主体框包含多深度、
    # 或焦距建议置信度过低。此时要求执行小范围验证帧扫描。
    report = {
        "input": str(image_path),
        "depth_model": depth_model,
        "detector_model": detector_model,
        "subject_count": len(ranked),
        "selected_subject_id": selected.subject.subject_id if selected else None,
        "initial_focus_zone": selected.focus_zone if selected else None,
        "requires_focus_sweep": (
            selected is None
            or selected.depth.multi_depth_ambiguity
            or selected.focus_confidence < 0.45
        ),
        "subjects": [item.to_dict() for item in ranked],
        "limitations": [
            "focus_zone is a relative near/mid/far recommendation, not a lens motor position",
            "camera and lens calibration plus a validation focus sweep are required for physical autofocus",
        ],
    }
    report_path = output_dir / "focus_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth-guided AI subject selection for autofocus.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/autofocus.json"))
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--detector-model", default="hustvl/yolos-tiny")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_frame(
        args.image,
        args.output,
        config_path=args.config,
        depth_model=args.depth_model,
        detector_model=args.detector_model,
        device=args.device,
    )
    print(f"Wrote focus analysis: {report}")


if __name__ == "__main__":
    main()
