from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class DepthEstimator:
    """Thin adapter around Hugging Face depth-estimation models."""

    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError("Install the project dependencies before running inference.") from error

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self._torch = torch
        # Pin the processor behavior so a future Transformers default change
        # does not silently alter an existing experiment.
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=False)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        """Predict a float32 relative-depth map at the original image size."""
        torch = self._torch
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted = outputs.predicted_depth.unsqueeze(1)
            resized = torch.nn.functional.interpolate(
                predicted,
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )
        depth = resized.squeeze().detach().float().cpu().numpy()
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise RuntimeError("model returned an invalid depth map")
        return depth.astype(np.float32, copy=False)
