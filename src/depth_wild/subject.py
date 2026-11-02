from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class SubjectCandidate:
    """AI 检测到的一个候选对焦主体。

    box 使用像素坐标 (x1, y1, x2, y2)，左上角为坐标原点。
    confidence 只表示检测模型对类别与框的置信度，不表示对焦一定正确。
    """

    subject_id: str
    label: str
    confidence: float
    box: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """计算两个候选框的交并比，用于识别重复检测结果。

    IoU 越接近 1，表示两个框覆盖的区域越相似。这里不使用图像内容，
    只根据几何重叠判断，因此它是一个轻量、可复现的后处理步骤。
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0 else 0.0


def suppress_duplicate_candidates(
    candidates: list[SubjectCandidate],
    *,
    iou_threshold: float = 0.55,
) -> list[SubjectCandidate]:
    """删除同类别且高度重叠的重复候选框。

    YOLOS/DETR 类检测器可能为同一个沙发或人物输出多个相近框。如果全部进入
    主体排序，同一物体会重复占据候选列表并干扰自动对焦。因此先按置信度从高
    到低保留候选，再抑制与已保留同类别框高度重叠的结果。

    不同类别暂不互相抑制，因为一个区域可能确实同时包含人和椅子；跨类别冲突
    将来可结合实例分割或主体跟踪进一步判断。
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    kept: list[SubjectCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            candidate.label == existing.label
            and box_iou(candidate.box, existing.box) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)

    # 去重后重新编号，使报告中的 subject_id 连续且便于人工检查。
    return [
        SubjectCandidate(
            subject_id=f"subject-{index + 1}",
            label=candidate.label,
            confidence=candidate.confidence,
            box=candidate.box,
        )
        for index, candidate in enumerate(kept)
    ]


class SubjectDetector:
    """第一版自动主体识别基线，使用 COCO 类别的目标检测模型。

    该类只负责产生候选主体，不负责决定最终对焦对象。最终排序由 focus.py
    综合类别偏好、构图和深度特征完成。后续可替换为人眼/鸟眼专用检测器。
    """

    def __init__(
        self,
        model_id: str = "hustvl/yolos-tiny",
        device: str = "auto",
        threshold: float = 0.55,
        nms_iou_threshold: float = 0.55,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as error:
            raise RuntimeError("Install the project dependencies before subject detection.") from error

        if device == "auto":
            # 当前 Windows 电脑没有 CUDA 时自动退回 CPU。
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.threshold = threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForObjectDetection.from_pretrained(model_id).to(device).eval()

    def detect(self, image: Image.Image) -> list[SubjectCandidate]:
        """检测图片中的候选主体，并把检测框恢复到原图像素尺寸。"""
        torch = self._torch
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        # Hugging Face 模型内部会缩放图片；这里将输出框映射回原始图像尺寸。
        result = self._processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

        candidates: list[SubjectCandidate] = []
        for index, (score, label_id, box) in enumerate(
            zip(result["scores"], result["labels"], result["boxes"])
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in box.tolist())
            label = self._model.config.id2label[int(label_id)]
            # 对检测框做边界裁剪，避免后续从深度图切片时出现越界。
            candidates.append(
                SubjectCandidate(
                    subject_id=f"subject-{index + 1}",
                    label=label.lower(),
                    confidence=float(score),
                    box=(
                        max(0, x1),
                        max(0, y1),
                        min(image.width, x2),
                        min(image.height, y2),
                    ),
                )
            )
        return suppress_duplicate_candidates(
            candidates,
            iou_threshold=self.nms_iou_threshold,
        )
