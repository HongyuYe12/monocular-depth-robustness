import numpy as np
from PIL import Image

from depth_wild.perturb import apply_perturbation, deterministic_seed


def test_perturbations_preserve_size_and_mode():
    image = Image.new("RGB", (32, 24), (80, 120, 160))
    for kind, value in [("brightness", 0.5), ("blur", 2.0), ("noise", 15.0), ("resolution", 0.5)]:
        result = apply_perturbation(image, kind, value, seed=7)
        assert result.size == image.size
        assert result.mode == "RGB"


def test_noise_is_deterministic():
    image = Image.new("RGB", (16, 16), (128, 128, 128))
    first = np.asarray(apply_perturbation(image, "noise", 20.0, seed=42))
    second = np.asarray(apply_perturbation(image, "noise", 20.0, seed=42))
    assert np.array_equal(first, second)


def test_stable_seed_changes_by_image():
    assert deterministic_seed(2026, "a", 15.0) == deterministic_seed(2026, "a", 15.0)
    assert deterministic_seed(2026, "a", 15.0) != deterministic_seed(2026, "b", 15.0)
