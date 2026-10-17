from __future__ import annotations

import argparse

from PIL import Image

from .infer import colorize_depth
from .metrics import robustness_metrics
from .model import DepthEstimator
from .perturb import apply_perturbation


def build_demo(model_id: str, device: str):
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Install the demo dependency with: pip install -e .[demo]") from error

    estimator = DepthEstimator(model_id=model_id, device=device)

    def run(image: Image.Image, perturbation: str, value: float):
        if image is None:
            raise gr.Error("Please upload an image.")
        original = image.convert("RGB")
        effective_value = 0.0 if perturbation == "original" else float(value)
        changed = apply_perturbation(original, perturbation, effective_value, seed=2026)
        original_depth = estimator.predict(original)
        changed_depth = estimator.predict(changed)
        metrics = robustness_metrics(changed_depth, original_depth, seed=2026)
        return changed, colorize_depth(original_depth), colorize_depth(changed_depth), metrics

    with gr.Blocks(title="Monocular Depth Robustness Demo") as app:
        gr.Markdown("# Monocular Depth Robustness Demo\nCompare relative-depth predictions before and after an image perturbation. Outputs are not metric distance.")
        with gr.Row():
            image = gr.Image(type="pil", label="Input image")
            changed = gr.Image(label="Perturbed image")
        with gr.Row():
            perturbation = gr.Dropdown(["original", "brightness", "blur", "noise", "resolution"], value="brightness", label="Perturbation")
            value = gr.Number(value=0.7, label="Value (see README for valid ranges)")
        run_button = gr.Button("Run depth comparison", variant="primary")
        with gr.Row():
            original_depth = gr.Image(label="Original depth")
            changed_depth = gr.Image(label="Perturbed depth")
        metrics = gr.JSON(label="Stability metrics")
        run_button.click(run, [image, perturbation, value], [changed, original_depth, changed_depth, metrics])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the interactive robustness demo.")
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_demo(args.model, args.device).launch(share=args.share)


if __name__ == "__main__":
    main()
