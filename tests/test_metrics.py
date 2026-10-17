import numpy as np

from depth_wild.metrics import affine_nrmse, edge_f1, order_consistency, spearman_correlation


def test_affine_nrmse_ignores_scale_and_shift():
    reference = np.arange(100, dtype=np.float64).reshape(10, 10)
    prediction = reference * 3.5 - 17.0
    assert affine_nrmse(prediction, reference) < 1e-10


def test_rank_and_order_are_perfect_for_monotonic_transform():
    reference = np.arange(400, dtype=np.float64).reshape(20, 20)
    prediction = reference * 2.0 + 5.0
    assert spearman_correlation(prediction, reference) > 0.999999
    assert order_consistency(prediction, reference, pairs=5000, seed=3) == 1.0


def test_order_detects_reversal():
    reference = np.arange(400, dtype=np.float64).reshape(20, 20)
    prediction = -reference
    assert order_consistency(prediction, reference, pairs=5000, seed=3) == 0.0


def test_edge_f1_identity():
    reference = np.zeros((20, 20), dtype=np.float64)
    reference[:, 10:] = 1.0
    assert edge_f1(reference, reference) == 1.0
