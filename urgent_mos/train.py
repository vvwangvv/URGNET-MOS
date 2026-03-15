#!/usr/bin/env python3
"""check scripts/train.sh for usage"""
from __future__ import annotations

from pathlib import Path
import json

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs/is27", config_name="train")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    seed = cfg.get("seed", 42)
    Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)

    config_str = json.dumps(OmegaConf.to_container(cfg, resolve=True))
    train_loader = instantiate(cfg.dataloader, data=cfg.train_data, is_train=True)
    cv_loader = instantiate(cfg.dataloader, data=cfg.cv_data, is_train=False)
    trainer = instantiate(cfg.trainer, config_str=config_str)
    trainer.train(train_loader, cv_loader, seed=seed)


if __name__ == "__main__":
    main()
