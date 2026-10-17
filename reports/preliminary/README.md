# Preliminary Two-Model Comparison

This report validates the end-to-end comparison pipeline. It is **not** a benchmark claim.

## Setup

- Images: 6 self-collected or licensed images
- Models: Depth Anything V2 Small and Intel DPT Hybrid MiDaS
- Perturbations: brightness, Gaussian blur, Gaussian noise, and downsample/upsample resolution loss
- Samples per plotted condition: 6
- Metrics: prediction stability relative to each model's own clean-image prediction
- Ground truth depth: unavailable

The results therefore describe robustness consistency, not absolute depth accuracy.

## Early observations

- Strong blur (`radius=4`) caused the largest structural degradation for both models.
- Depth Anything V2 showed lower affine-aligned error than MiDaS under the tested brightness and blur changes.
- MiDaS retained slightly stronger edge overlap under mild resolution loss in this small sample.
- Both models were comparatively stable under mild perturbations.

These observations are hypotheses for the larger experiment, not final conclusions. The next milestone is a labeled 30–50 image pilot, followed by a 100–200 image dataset and a public relative-depth benchmark such as DA-2K.

## Files

- `model_summary.csv`: aggregated results by model, perturbation, and value
- `scene_summary.csv`: scene-level aggregation; currently unlabeled until metadata is completed
- `compare_*.png`: comparison curves for four robustness metrics

No source images, model weights, raw predictions, or per-image filenames are published in this directory.
