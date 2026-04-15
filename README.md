# vehicle-keypoints

[![CI](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)

Production-grade vehicle keypoint detection (14 anatomical car keypoints, CarFusion).

**Russian version:** [README.ru.md](README.ru.md)

## Task

Task type: `keypoints` · Framework: `pytorch`.

## Dataset

Document dataset source, size, splits. Link to Kaggle / HF dataset page.

## Results

Fill in after training. Include metrics table with main model vs baseline.

| Model | Metric 1 | Metric 2 |
|---|---|---|
| Main | — | — |
| Baseline | — | — |

## Quick Start

```bash
uv sync --all-groups
make data
make train
make evaluate
make serve
docker compose up
```

## Project Structure

```
src/vehicle_keypoints/
├── data/
├── models/
├── training/
├── evaluation/
├── inference/
├── serving/
└── utils/
```

## License

MIT — see [LICENSE](LICENSE).
