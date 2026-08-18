from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "files"))
from utils import (  # noqa: E402
    accuracy,
    baseline_predictions,
    circle_iou,
    fit_circle,
    generate_dataset,
    noisy_circle,
)


def test_generated_data_is_deterministic_and_has_expected_shapes():
    first_images, first_labels = generate_dataset(3, size=32, max_radius=10, seed=11)
    second_images, second_labels = generate_dataset(3, size=32, max_radius=10, seed=11)

    assert first_images.shape == (3, 32, 32, 1)
    assert first_labels.shape == (3, 3)
    np.testing.assert_array_equal(first_images, second_images)
    np.testing.assert_array_equal(first_labels, second_labels)


def test_circle_parameters_are_valid_and_image_is_bounded():
    labels, image = noisy_circle(size=32, max_radius=10, noise=0.2, rng=np.random.default_rng(2))
    row, col, radius = labels

    assert radius <= row <= 31 - radius
    assert radius <= col <= 31 - radius
    assert image.min() >= 0
    assert image.max() <= 1


def test_circle_iou_edge_cases():
    assert circle_iou([10, 10, 4], [10, 10, 4]) == pytest.approx(1.0)
    assert circle_iou([0, 0, 1], [10, 10, 1]) == pytest.approx(0.0)
    assert circle_iou([0, 0, 4], [0, 0, 2]) == pytest.approx(0.25)


def test_accuracy_requires_matching_circle_arrays():
    assert accuracy(np.array([[10, 10, 4]]), np.array([[10, 10, 4]])) == 1.0
    with pytest.raises(ValueError):
        accuracy(np.array([[10, 10, 4]]), np.array([[10, 10]]))


def test_threshold_baseline_recovers_synthetic_circle():
    labels, image = noisy_circle(size=64, max_radius=20, rng=np.random.default_rng(8))
    prediction = fit_circle(image)

    assert circle_iou(prediction, labels) > 0.98
    batch_prediction = baseline_predictions(image[None, :, :, None])
    assert accuracy(batch_prediction, labels[None, :]) == 1.0


def test_circle_fit_requires_sufficient_signal():
    with pytest.raises(ValueError, match="at least three bright pixels"):
        fit_circle(np.zeros((8, 8)))
