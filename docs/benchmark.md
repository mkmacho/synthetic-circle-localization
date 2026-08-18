# Reference benchmark

This result measures the updated CNN on the synthetic generator in this
repository. It is not a real-world circle-detection result.

## Design

- TensorFlow 2.18.1 in the supplied Linux ARM64 Docker image.
- A fresh, deterministic training set of 4,000 images per seed.
- A separate deterministic held-out set of 1,000 images per seed.
- 64×64 grayscale images, maximum radius 20 pixels, and uniform background
  noise in [0, 0.15].
- Ten training epochs; accuracy is the share of examples whose predicted disk
  has intersection over union (IoU) of at least 0.70 with the target disk.

The model was never initialized from the analytical baseline or from any
previous model. The baseline uses thresholded bright perimeter pixels and is
therefore intentionally advantaged by this synthetic generator.

## Results

| Seed | CNN IoU@0.70 | Threshold baseline IoU@0.70 |
| ---: | ---: | ---: |
| 7 | 86.2% | 100.0% |
| 8 | 86.4% | 100.0% |
| 9 | 87.9% | 100.0% |
| **Mean** | **86.8%** | **100.0%** |

The CNN range was 86.2%–87.9%. These results use the exact configuration in
`scripts/run_benchmark.sh`; generated models, metrics, and figures are ignored
by Git and can be reproduced locally.
