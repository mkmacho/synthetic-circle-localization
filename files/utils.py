"""Synthetic-circle data generation and evaluation helpers."""

from __future__ import annotations

import numpy as np


def draw_circle(image: np.ndarray, row: float, col: float, radius: float) -> None:
    """Draw a one-pixel circular perimeter into a square, single-channel image."""
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")

    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    distance = np.sqrt((y - row) ** 2 + (x - col) ** 2)
    image[np.abs(distance - radius) <= 0.75] = 1.0


def noisy_circle(
    size: int = 128,
    max_radius: int = 40,
    noise: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``([row, col, radius], image)`` for one noisy circle.

    The circle is kept fully in-frame, so radius and center have an unambiguous
    target. Image values are clipped to the [0, 1] range.
    """
    if size < 8:
        raise ValueError("size must be at least 8")
    if not 2 <= max_radius < size / 2:
        raise ValueError("max_radius must be between 2 and size / 2")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = rng or np.random.default_rng()
    radius = int(rng.integers(2, max_radius + 1))
    row = int(rng.integers(radius, size - radius))
    col = int(rng.integers(radius, size - radius))
    image = np.clip(rng.uniform(0, noise, size=(size, size)), 0, 1).astype(np.float32)
    draw_circle(image, row, col, radius)
    return np.array([row, col, radius], dtype=np.float32), image


def generate_dataset(
    n_samples: int,
    size: int = 128,
    max_radius: int = 40,
    noise: float = 0.15,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create images of shape ``(n, size, size, 1)`` and pixel-space labels."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    labels = np.empty((n_samples, 3), dtype=np.float32)
    images = np.empty((n_samples, size, size, 1), dtype=np.float32)
    for index in range(n_samples):
        labels[index], images[index, :, :, 0] = noisy_circle(
            size=size, max_radius=max_radius, noise=noise, rng=rng
        )
    return images, labels


def circle_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return analytic IoU for two disks parameterized as row, col, radius."""
    row0, col0, radius0 = np.asarray(first, dtype=float)
    row1, col1, radius1 = np.asarray(second, dtype=float)
    if radius0 <= 0 or radius1 <= 0:
        return 0.0

    distance = float(np.hypot(row0 - row1, col0 - col1))
    if distance >= radius0 + radius1:
        return 0.0
    if distance <= abs(radius0 - radius1):
        overlap = np.pi * min(radius0, radius1) ** 2
    else:
        alpha = np.arccos(
            np.clip((distance**2 + radius0**2 - radius1**2) / (2 * distance * radius0), -1, 1)
        )
        beta = np.arccos(
            np.clip((distance**2 + radius1**2 - radius0**2) / (2 * distance * radius1), -1, 1)
        )
        overlap = (
            radius0**2 * alpha
            + radius1**2 * beta
            - 0.5
            * np.sqrt(
                max(
                    0.0,
                    (-distance + radius0 + radius1)
                    * (distance + radius0 - radius1)
                    * (distance - radius0 + radius1)
                    * (distance + radius0 + radius1),
                )
            )
        )
    union = np.pi * radius0**2 + np.pi * radius1**2 - overlap
    return float(overlap / union)


def accuracy(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.7) -> float:
    """Fraction of predicted disks whose IoU with their target exceeds a threshold."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape or predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("predictions and labels must both have shape (n, 3)")
    return float(np.mean([circle_iou(pred, label) >= threshold for pred, label in zip(predictions, labels)]))


def fit_circle(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Fit a circle to bright perimeter pixels with linear least squares.

    This deliberately simple baseline is appropriate only for this synthetic
    data distribution, where the circle is brighter than the background noise.
    It is included to make the CNN's performance interpretable, not as a
    general-purpose circle detector.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must have shape (height, width)")
    rows, cols = np.nonzero(image >= threshold)
    if len(rows) < 3:
        raise ValueError("at least three bright pixels are required to fit a circle")

    design = np.column_stack((2 * rows, 2 * cols, np.ones(len(rows))))
    target = rows**2 + cols**2
    row, col, offset = np.linalg.lstsq(design, target, rcond=None)[0]
    radius_squared = offset + row**2 + col**2
    if radius_squared <= 0:
        raise ValueError("the selected pixels do not define a valid circle")
    return np.array([row, col, np.sqrt(radius_squared)], dtype=np.float32)


def baseline_predictions(images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict circle parameters for a batch using the threshold-fit baseline."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 1:
        raise ValueError("images must have shape (n, height, width, 1)")
    return np.stack([fit_circle(image[:, :, 0], threshold) for image in images])
