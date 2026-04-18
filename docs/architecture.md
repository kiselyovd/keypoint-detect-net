# Architecture

Two independent branches that share the same CarFusion COCO export but diverge at the dataset layer: YOLO consumes a processed YOLO-layout tree via `data.yaml`, while ViTPose reads the COCO JSON directly and renders Gaussian heatmaps per visible keypoint.

## Data flow

```mermaid
flowchart LR
  Src[CarFusion COCO<br/>D:/ProjectsData/Car Key Point] --> Sync[scripts/sync_data.sh]
  Sync --> Raw["data/raw/{annotations,train,test}<br/>gitignored"]
  Raw --> Prep[data/prepare.py<br/>scene-level 90/10 split]
  Prep --> Proc["data/processed/{images,labels}<br/>YOLO layout + data.yaml"]

  Proc --> YOLOTrain[training/train.py<br/>Ultralytics YOLO26-pose]
  YOLOTrain --> MainArt["artifacts/&lt;run&gt;/weights/best.pt"]
  MainArt --> Publish[scripts/publish_to_hf.py]
  Publish --> HF[HF Hub<br/>kiselyovd/vehicle-keypoints]

  Raw --> COCODS[CocoKeypointsDataset<br/>top-down crops + Gaussian heatmaps]
  COCODS --> ViTTrain[training/train_vitpose.py<br/>Lightning + Hydra + MLflow]
  ViTTrain --> BaseArt[artifacts/baseline/checkpoints/best.ckpt]
  BaseArt --> Publish
  Publish --> HFBase[HF Hub: baseline/ subdir]

  MainArt --> Eval[evaluation/evaluate.py<br/>OKS-mAP + PCK]
  BaseArt --> Eval
  MainArt --> API[FastAPI /detect]
```

## Model-choice rationale

**Main — YOLO26-pose.** Ultralytics' YOLO26-pose is the natural main for this task: it is a single-shot detector that jointly regresses bounding boxes, objectness, and a configurable number of keypoints per instance. The `kpt_shape=[14, 3]` hyperparameter cleanly accommodates non-human keypoint classes without any architectural surgery. Throughput on an RTX 3080 is an order of magnitude higher than top-down alternatives (no explicit crop step), and the produced `.pt` checkpoint plugs straight into the Ultralytics inference CLI, ONNX exporter, and Hub publishing flow. These operational wins matter as much as the accuracy numbers.

**Baseline — ViTPose-S.** ViTPose is the academic reference implementation for transformer-based 2D pose estimation. Using the small variant (`ViTPose-S`, ~22 M params) pretrained on COCO human 17-keypoint gives us a concrete transfer-learning story: we replace the keypoint head with a fresh 14-channel deconv head and fine-tune. This provides a meaningful second number on the leaderboard and demonstrates that the repo is not locked into a single framework — the same evaluation protocol runs against both branches.

**Metrics — OKS-mAP + PCK.** Object Keypoint Similarity mAP is the COCO-standard metric for keypoint tasks: it accounts for per-keypoint sigmas (tolerance) and instance scale, which makes comparison against published baselines meaningful. We pair it with PCK@0.05 (Percentage of Correct Keypoints within 5% of the bounding-box diagonal), which is intuitive to read at a glance — "of all visible keypoints, how many did we get right?" — and is robust to the absence of validated per-keypoint sigmas for the car class.
