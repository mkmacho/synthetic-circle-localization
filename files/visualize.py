"""Visualization helpers for saved model predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_prediction_figure(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    count: int = 9,
) -> None:
    """Save a grid with target and predicted circle perimeters overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    count = min(count, len(images))
    if count < 1:
        raise ValueError("at least one image is required")
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(images[index, :, :, 0], cmap="gray", vmin=0, vmax=1)
        target_row, target_col, target_radius = labels[index]
        predicted_row, predicted_col, predicted_radius = predictions[index]
        axis.add_patch(Circle((target_col, target_row), target_radius, fill=False, color="#31c48d", lw=2))
        axis.add_patch(Circle((predicted_col, predicted_row), predicted_radius, fill=False, color="#ef4444", lw=2))
        axis.set_title("target: green; CNN: red")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
