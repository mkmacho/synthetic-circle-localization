"""TensorFlow 2 model for localizing synthetic circles."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def build_model(image_size: int) -> tf.keras.Model:
    """Build a small convolutional regressor with normalized circle targets."""
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="image")
    x = inputs
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    # Flattening deliberately preserves spatial position. A global average pool
    # would make this regressor largely translation-invariant, despite location
    # being the task's primary target.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    outputs = tf.keras.layers.Dense(3, activation="sigmoid", name="circle_parameters")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="circle_localizer")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
    return model


def normalize_labels(labels: np.ndarray, image_size: int) -> np.ndarray:
    """Scale row, column, and radius labels into the [0, 1] range."""
    scale = np.array([image_size - 1, image_size - 1, image_size / 2], dtype=np.float32)
    return np.asarray(labels, dtype=np.float32) / scale


def denormalize_labels(labels: np.ndarray, image_size: int) -> np.ndarray:
    """Convert normalized model outputs to pixel-space circle parameters."""
    scale = np.array([image_size - 1, image_size - 1, image_size / 2], dtype=np.float32)
    return np.asarray(labels, dtype=np.float32) * scale
