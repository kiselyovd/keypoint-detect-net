# KeypointDetectNet

Vehicle keypoint detection using Keypoint R-CNN with ResNet-50 FPN backbone. Detects and localizes 14 structural keypoints per vehicle instance in images.

## Overview

This project implements a keypoint detection pipeline for vehicles using the [CarFusion](https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html) dataset. It includes a data conversion tool (CarFusion → COCO format) and a training pipeline built on PyTorch's Keypoint R-CNN.

## Dataset

**CarFusion** (CMU) — multi-view car images with annotated keypoints.

Each vehicle instance is annotated with:
- 14 keypoints (front/rear corners, doors, windows)
- Bounding boxes
- Visibility flags (visible / occluded / not visible)
- Segmentation masks

<p align="center">
  <img src="assets/cars.jpg" width="70%" alt="CarFusion dataset sample" />
</p>

### COCO Annotation Format

```json
{
  "image_id": 101100001,
  "category_id": 1,
  "bbox": [806, 497, 879, 268],
  "num_keypoints": 12,
  "keypoints": [913, 708, 1, 989, 734, 1, ...]
}
```

## Model

**Architecture**: Keypoint R-CNN with ResNet-50 + Feature Pyramid Network  
**Classes**: 2 (background + vehicle)  
**Optimizer**: SGD (lr=0.001, momentum=0.9, weight_decay=0.5)  
**Scheduler**: MultiStepLR (γ=0.1, step every 3 epochs)

## Project Structure

```
├── scripts/
│   └── carfusion2coco.py      # CarFusion to COCO format converter
├── notebooks/
│   └── training.ipynb         # Training and inference notebook
└── assets/
    └── cars.jpg               # Dataset sample
```

## Data Conversion

Convert CarFusion annotations to COCO keypoint format:

```bash
python scripts/carfusion2coco.py \
    --path_dir ./data \
    --label_dir labels \
    --image_dir images \
    --output_dir ./coco_annotations \
    --output_filename annotations.json
```

## Training

```python
from torchvision.models.detection import keypointrcnn_resnet50_fpn

model = keypointrcnn_resnet50_fpn(weights="DEFAULT", num_keypoints=14, num_classes=2)
```

## Pre-trained Weights

Trained weights are available on [Google Drive](https://drive.google.com/file/d/1457APbaetA9OuRV3Icm_MsC4majXXgUJ/view?usp=sharing).

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision
- shapely
- numpy
- tqdm

## References

- [CarFusion Dataset](https://www.cs.cmu.edu/~ILIM/projects/IM/CarFusion/cvpr2018/index.html)
- [carfusion_to_coco](https://github.com/dineshreddy91/carfusion_to_coco) — original conversion script

## License

MIT
