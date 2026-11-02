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
        return candidates
