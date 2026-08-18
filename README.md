# Find Circle

A reproducible computer-vision experiment: train a convolutional regressor to
estimate the center and radius of a noisy, synthetic circle. Images are
generated deterministically at runtime, so the project needs no external data
or pretrained checkpoints.

The benchmark is deliberately simple. Alongside the CNN, the program evaluates
a threshold-and-least-squares baseline. That baseline exploits the synthetic
data generator's bright, clean perimeter, so it should be treated as an
interpretability check rather than a general-purpose detector. The CNN is the
learning exercise; neither approach is intended for real-world images.

## Quick start

Requires Python 3.10–3.12 and a TensorFlow-compatible platform.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python files/main.py
```

The default configuration uses seed `7`, 8,000 generated training samples, and
1,000 held-out samples. It writes these reproducible artifacts:

- `artifacts/circle_localizer.keras` — saved CNN;
- `artifacts/metrics.json` — configuration and held-out CNN/baseline scores;
- `artifacts/predictions.png` — target (green) and CNN prediction (red).

For a quick smoke run:

```sh
python files/main.py --epochs 1 --train-samples 256 --eval-samples 64
```

To score an existing model without re-training:

```sh
python files/main.py --evaluate-only
```

## Docker

```sh
./run.sh --epochs 1 --train-samples 256 --eval-samples 64
```

The generated artifacts are persisted under `artifacts/` on the host.

## Reference result

Across three independent deterministic runs, the CNN achieved a mean **86.8%
IoU@0.70** (range 86.2%–87.9%) on the synthetic held-out distribution. See the
[benchmark protocol and full results](docs/benchmark.md). Reproduce the three
runs locally with:

```sh
./scripts/run_benchmark.sh
```

## Verification

```sh
python -m pytest
```

The tests cover deterministic generation, valid bounds, analytic IoU edge
cases, and exact recovery by the synthetic-data baseline. GitHub Actions runs
this suite and a source-compilation check on Python 3.12 for every push and
pull request.

## Project layout

- `files/utils.py` — generator, IoU metric, and transparent baseline.
- `files/conv.py` — TensorFlow 2/Keras spatial regressor.
- `files/main.py` — reproducible train/evaluate workflow and metrics output.
- `files/visualize.py` — saved prediction grid.
- `tests/` — dependency-light checks.

## License

This project is available under the [MIT License](LICENSE).
