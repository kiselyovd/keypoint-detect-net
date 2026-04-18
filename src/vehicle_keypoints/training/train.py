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

    import ultralytics
    from ultralytics import YOLO

    # Force Ultralytics to use our configured output dir (its global settings would otherwise win).
    out_abs = str(Path(cfg.trainer.output_dir).resolve())
    ultralytics.settings.update({"runs_dir": out_abs})

    model = YOLO(cfg.model.name + "-pose.pt")
    model.train(
        data=cfg.data.yaml_path,
        epochs=cfg.trainer.max_epochs,
        imgsz=cfg.data.image_size,
        batch=cfg.data.batch_size,
        project=out_abs,
        name=cfg.experiment_name,
        workers=0,  # Windows multiprocessing hangs with workers>0 + ultralytics spawn
        patience=max(5, int(cfg.trainer.max_epochs * 0.3)),  # early stop if no improvement
    )
    log.info("train.done")
    return


if __name__ == "__main__":
    main()
