"""Training entrypoint (Hydra-powered)."""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from ..models import build_model
    from ultralytics import YOLO

    model = YOLO(cfg.model.name + "-pose.pt")
    model.train(
        data=cfg.data.yaml_path,
        epochs=cfg.trainer.max_epochs,
        imgsz=cfg.data.image_size,
        batch=cfg.data.batch_size,
        project=cfg.trainer.output_dir,
        name=cfg.experiment_name,
    )
    log.info("train.done")
    return


if __name__ == "__main__":
    main()
