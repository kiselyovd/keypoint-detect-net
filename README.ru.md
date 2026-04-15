# vehicle-keypoints

[![CI](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml/badge.svg)](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)

Production-grade vehicle keypoint detection (14 anatomical car keypoints, CarFusion).

**English:** [README.md](README.md)

## Задача

Тип задачи: `keypoints` · Фреймворк: `pytorch`.

## Датасет

Укажите источник датасета, размер, разбиение. Ссылка на Kaggle / HF.

## Результаты

Заполняется после обучения. Таблица метрик: основная модель vs baseline.

| Модель | Метрика 1 | Метрика 2 |
|---|---|---|
| Основная | — | — |
| Baseline | — | — |

## Быстрый старт

```bash
uv sync --all-groups
make data
make train
make evaluate
make serve
docker compose up
```

## Структура проекта

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

## Лицензия

MIT — см. [LICENSE](LICENSE).
