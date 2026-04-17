# vehicle-keypoints

Production-уровневая оценка позы автомобиля по 14 ключевым точкам на CarFusion — YOLO26-pose (основная) + ViTPose-S (baseline).

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/vehicle-keypoints/ci.yml?branch=main&label=CI&style=for-the-badge&logo=github)](https://github.com/kiselyovd/vehicle-keypoints/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://kiselyovd.github.io/vehicle-keypoints/)
[![codecov](https://img.shields.io/codecov/c/github/kiselyovd/vehicle-keypoints?style=for-the-badge&logo=codecov&logoColor=white)](https://codecov.io/gh/kiselyovd/vehicle-keypoints)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Model](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/vehicle-keypoints)

Комплексный инструментарий для детекции автомобилей и регрессии **14 анатомических ключевых точек** (колёса, фары, зеркала, углы крыши) на датасете CarFusion (CMU). Основная модель — одностадийный детектор **YOLO26-pose** (Ultralytics); baseline — top-down **ViTPose-S**, обёрнутый в PyTorch Lightning и получающий кропы от YOLO-детектора автомобилей. Стек настраивается через Hydra, обучается из CLI, оценивается по OKS-mAP + PCK@0.05, публикуется через FastAPI + Docker, документируется MkDocs Material и распространяется через Hugging Face Hub. Только для исследований и обучения — **не является основным сенсором для автономного вождения**.

📖 [English docs](https://kiselyovd.github.io/vehicle-keypoints/) • 🇬🇧 [English README](README.md) • 🤗 [HF Hub модель](https://huggingface.co/kiselyovd/vehicle-keypoints)

## Датасет

[CarFusion](http://www.cs.cmu.edu/~mvo/index_files/CarFusion.html) — многокамерный датасет дорожных сцен от Carnegie Mellon University (Dinesh Reddy, Minh Vo, Srinivasa Narasimhan, CVPR 2018), содержащий **14 ключевых точек на каждом автомобиле**. Исходный релиз предварительно конвертируется в формат COCO-keypoints через `scripts/convert_carfusion_to_coco.py`, а `src/vehicle_keypoints/data/prepare.py` выполняет **разбиение 90/10 на уровне сцен** среди 8 обучающих сцен (val = `car_craig2`, остальные 7 — в train), так что ни одна сцена не попадает сразу в две выборки.

Итоговое количество изображений: **16 713 train / 3 474 val / 12 761 test**. CarFusion принадлежит CMU и распространяется под собственной исследовательской лицензией.

## Результаты

Метрики на тестовой выборке после полного обучения (будут заполнены из `reports/metrics.json` по итогам запуска v0.1.0):

| Модель | OKS-mAP | OKS-mAP50 | PCK@0.05 |
|---|---|---|---|
| **YOLO26-pose** (основная) | — | — | — |
| ViTPose-S (baseline) | — | — | — |

Метрики считаются pycocotools + собственной реализацией PCK@0.05 на тестовой части CarFusion (12 761 изображение). Будут заполнены после тренировочного запуска v0.1.0.

## Быстрый старт

```bash
uv sync --all-groups
bash scripts/sync_data.sh "D:/ProjectsData/Car Key Point/datasets/carfusion"
uv run python -m vehicle_keypoints.data.prepare --raw data/raw --out data/processed
make train     # main YOLO26-pose (~90 min RTX 3080)
make evaluate  # OKS-mAP + PCK on test
make serve     # FastAPI on :8000
```

## Полные команды обучения

**Основная — YOLO26-pose:**

```bash
uv run python -m vehicle_keypoints.training.train +experiment=sota trainer.max_epochs=100
```

**Baseline — ViTPose-S (top-down, Lightning):**

```bash
uv run python -m vehicle_keypoints.training.train_vitpose model=baseline trainer.max_epochs=30
```

## Инференс

```python
from huggingface_hub import hf_hub_download
from vehicle_keypoints.inference.predict import Detector

ckpt = hf_hub_download(repo_id="kiselyovd/vehicle-keypoints", filename="weights.pt")
det = Detector.from_checkpoint(ckpt)
detections = det.predict("car.jpg")
for d in detections:
    print(d["bbox"], d["score"], len(d["keypoints"]))
```

Каждая детекция — словарь с рамкой `bbox` в формате xyxy, скором детектора `score` и списком из 14 точек `(x, y, visibility)`, соответствующих скелету CarFusion.

## Сервинг

```bash
docker compose up api
```

JSON-ответ (ключевые точки + рамки):

```bash
curl -X POST -F file=@car.jpg http://localhost:8000/detect
```

Overlay-PNG (скелет, отрисованный поверх входного изображения):

```bash
curl -X POST -F file=@car.jpg "http://localhost:8000/detect?overlay=true" -o overlay.png
```

## Структура проекта

```
vehicle-keypoints/
├── configs/              # Hydra
├── data/                 # raw + processed + sample (CI)
├── docs/                 # MkDocs Material
├── scripts/              # convert_carfusion_to_coco.py, build_sample_data.py, publish_to_hf.py, ...
├── src/vehicle_keypoints/
│   ├── data/             # prepare.py, coco_dataset.py, datamodule.py
│   ├── models/           # factory.py, vitpose.py, lightning_module.py
│   ├── training/         # train.py (YOLO), train_vitpose.py (Lightning)
│   ├── evaluation/       # evaluate.py (OKS-mAP + PCK)
│   ├── inference/        # predict.py, overlay.py
│   └── serving/          # FastAPI /detect
└── tests/
```

## Назначение

Только для исследований и обучения в области компьютерного зрения. CarFusion — небольшой односценарный датасет, не отражающий всего разнообразия дорожных условий. **Не подходит в качестве основного сенсора для автономного вождения; не имеет сертификации по безопасности.**

## Лицензия

MIT — см. [LICENSE](LICENSE). Веса модели на HF Hub распространяются на тех же условиях; датасет CarFusion принадлежит CMU и подчиняется собственной исследовательской лицензии.
