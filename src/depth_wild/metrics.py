from __future__ import annotations

import numpy as np
from PIL import Image


def resize_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape == shape:
        return depth.astype(np.float64, copy=False)
    image = Image.fromarray(depth.astype(np.float32))
    resized = image.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float64)


def affine_align(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Least-squares align prediction to reference using scale and shift."""
    x = prediction.reshape(-1).astype(np.float64)
    y = reference.reshape(-1).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        raise ValueError("not enough finite pixels for alignment")
    design = np.column_stack((x[valid], np.ones(valid.sum(), dtype=np.float64)))
    scale, shift = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return prediction.astype(np.float64) * scale + shift


def affine_nrmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    aligned = affine_align(prediction, reference)
    valid = np.isfinite(aligned) & np.isfinite(reference)
    rmse = np.sqrt(np.mean((aligned[valid] - reference[valid]) ** 2))
    scale = np.std(reference[valid])
    return float(rmse / max(scale, 1e-12))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 50_000,
    seed: int = 0,
) -> float:
    x = prediction.reshape(-1).astype(np.float64)
    y = reference.reshape(-1).astype(np.float64)
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if valid.size < 2:
        return float("nan")
    if valid.size > max_samples:
        valid = np.random.default_rng(seed).choice(valid, max_samples, replace=False)
    rank_x = _average_ranks(x[valid])
    rank_y = _average_ranks(y[valid])
    if np.std(rank_x) < 1e-12 or np.std(rank_y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def order_consistency(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    pairs: int = 20_000,
    margin_fraction: float = 0.01,
    seed: int = 0,
) -> float:
    x = prediction.reshape(-1).astype(np.float64)
    y = reference.reshape(-1).astype(np.float64)
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if valid.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    first = rng.choice(valid, pairs, replace=True)
    second = rng.choice(valid, pairs, replace=True)
    ref_delta = y[first] - y[second]
    pred_delta = x[first] - x[second]
    margin = margin_fraction * max(float(np.ptp(y[valid])), 1e-12)
    informative = np.abs(ref_delta) > margin
    if not informative.any():
        return float("nan")
    return float(np.mean(np.sign(ref_delta[informative]) == np.sign(pred_delta[informative])))


def _edge_mask(depth: np.ndarray, quantile: float = 0.9) -> np.ndarray:
    array = depth.astype(np.float64)
    grad_y, grad_x = np.gradient(array)
    magnitude = np.hypot(grad_x, grad_y)
    finite_positive = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
    if finite_positive.size == 0:
        return np.zeros(depth.shape, dtype=bool)
    threshold = np.quantile(finite_positive, quantile)
    return magnitude >= threshold


def edge_f1(prediction: np.ndarray, reference: np.ndarray, *, quantile: float = 0.9) -> float:
    predicted_edges = _edge_mask(prediction, quantile)
    reference_edges = _edge_mask(reference, quantile)
    true_positive = np.logical_and(predicted_edges, reference_edges).sum()
    denominator = predicted_edges.sum() + reference_edges.sum()
    return float(2 * true_positive / denominator) if denominator else 1.0


def robustness_metrics(prediction: np.ndarray, reference: np.ndarray, *, seed: int = 0) -> dict[str, float]:
    prediction = resize_depth(prediction, reference.shape)
    return {
        "spearman": spearman_correlation(prediction, reference, seed=seed),
        "order_consistency": order_consistency(prediction, reference, seed=seed),
        "affine_nrmse": affine_nrmse(prediction, reference),
        "edge_f1": edge_f1(prediction, reference),
    }
