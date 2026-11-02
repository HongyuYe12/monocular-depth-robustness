from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .camera_input import load_camera_frame

# 深度估计负责把镜头带到合理区域，本文件负责在该区域附近用清晰度反馈微调。


def parse_box(value: str) -> tuple[int, int, int, int]:
    """解析命令行传入的主体框 x1,y1,x2,y2。"""
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("box must have positive width and height")
    return x1, y1, x2, y2


def subject_sharpness(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    """计算已选主体区域内的拉普拉斯方差，作为基础清晰度指标。

    分数越高通常表示边缘细节越丰富。它会受纹理、噪声和运动模糊影响，
    因此后续可替换为学习型清晰度模型，但当前方法适合作为可解释基线。
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    height, width = gray.shape
    x1, y1, x2, y2 = box
    region = gray[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if region.shape[0] < 3 or region.shape[1] < 3:
        raise ValueError("subject box is too small for focus validation")

    # 使用离散拉普拉斯算子提取高频边缘，再计算其方差。
    center = region[1:-1, 1:-1]
    laplacian = (
        -4 * center
        + region[:-2, 1:-1]
        + region[2:, 1:-1]
        + region[1:-1, :-2]
        + region[1:-1, 2:]
    )
    return float(np.var(laplacian))


def validate_focus_sweep(
    frames: list[Path],
    box: tuple[int, int, int, int],
    positions: list[float] | None = None,
) -> dict[str, object]:
    """从预测焦点附近的一组验证帧中选出主体最清晰的一帧。"""
    if not frames:
        raise ValueError("focus sweep contains no frames")
    if positions is not None and len(positions) != len(frames):
        raise ValueError("positions must match the number of frames")

    # 所有帧必须使用同一个主体框；真实视频模式还需要主体跟踪来更新框位置。
    scores = [subject_sharpness(load_camera_frame(frame), box) for frame in frames]
    best_index = int(np.argmax(scores))
    result: dict[str, object] = {
        "subject_box": list(box),
        "best_index": best_index,
        "best_frame": str(frames[best_index]),
        "scores": [
            {
                "index": index,
                "frame": str(frame),
                "sharpness": score,
                "focus_position": positions[index] if positions is not None else None,
            }
            for index, (frame, score) in enumerate(zip(frames, scores))
        ],
    }
    if positions is not None:
        result["best_focus_position"] = positions[best_index]
        # 默认传入顺序为近焦 → 当前预测 → 远焦，因此最优帧位于中心两侧时，
        # 可以给镜头控制层输出相对移动方向。
        middle = len(positions) // 2
        result["adjustment"] = (
            "toward-near" if best_index < middle else "toward-far" if best_index > middle else "hold"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Choose the sharpest validation frame inside a subject box.")
    parser.add_argument("--frames", type=Path, nargs="+", required=True)
    parser.add_argument("--box", type=parse_box, required=True, help="x1,y1,x2,y2")
    parser.add_argument("--positions", type=float, nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_focus_sweep(args.frames, args.box, args.positions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote focus validation: {args.output}")


if __name__ == "__main__":
    main()
