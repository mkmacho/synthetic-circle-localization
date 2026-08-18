"""Train or evaluate a CNN that localizes synthetic noisy circles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from conv import build_model, denormalize_labels, normalize_labels
from utils import accuracy, baseline_predictions, generate_dataset
from visualize import save_prediction_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/circle_localizer.keras"))
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--plot-path", type=Path, default=Path("artifacts/predictions.png"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-samples", type=int, default=8_000)
    parser.add_argument("--eval-samples", type=int, default=1_000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-radius", type=int, default=40)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot-samples", type=int, default=9)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def write_metrics(path: Path, metrics: dict[str, float | int]) -> None:
    """Persist machine-readable evaluation details alongside trained artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    eval_images, eval_labels = generate_dataset(
        args.eval_samples,
        args.image_size,
        args.max_radius,
        args.noise,
        seed=args.seed + 1,
    )
    if args.evaluate_only:
        model = tf.keras.models.load_model(args.model_path)
    else:
        train_images, train_labels = generate_dataset(
            args.train_samples,
            args.image_size,
            args.max_radius,
            args.noise,
            seed=args.seed,
        )
        model = build_model(args.image_size)
        model.fit(
            train_images,
            normalize_labels(train_labels, args.image_size),
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=64,
            verbose=2,
        )
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_path)

    predictions = denormalize_labels(model.predict(eval_images, verbose=0), args.image_size)
    cnn_accuracy = accuracy(predictions, eval_labels)
    baseline_accuracy = accuracy(baseline_predictions(eval_images), eval_labels)
    metrics = {
        "baseline_iou_at_0_70": baseline_accuracy,
        "cnn_iou_at_0_70": cnn_accuracy,
        "epochs": args.epochs,
        "evaluation_samples": args.eval_samples,
        "image_size": args.image_size,
        "max_radius": args.max_radius,
        "noise": args.noise,
        "seed": args.seed,
        "training_samples": args.train_samples,
    }
    write_metrics(args.metrics_path, metrics)
    if not args.no_plot:
        save_prediction_figure(eval_images, eval_labels, predictions, args.plot_path, args.plot_samples)

    print(f"CNN IoU@0.70 accuracy on {args.eval_samples:,} held-out samples: {cnn_accuracy:.1%}")
    print(f"Threshold baseline IoU@0.70 accuracy: {baseline_accuracy:.1%}")
    print(f"Metrics: {args.metrics_path}")
    if not args.no_plot:
        print(f"Visualization: {args.plot_path}")


if __name__ == "__main__":
    main()
