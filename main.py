import os
from pprint import pprint

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.utils.save_ckpt import save_and_push_best_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # ─────────── kiểm tra user ───────────
    if cfg.user is None:
        raise hydra.errors.HydraException(
            "Bạn phải truyền +user=<wandb_username> khi chạy!"
        )

    pprint(cfg)

    # ─────────── khởi tạo model & datamodule ───────────
    model = instantiate(cfg.experiment.model)
    datamodule = instantiate(cfg.experiment.datamodule)

    # ─────────── logger (wandb) ───────────
    logger = instantiate(cfg.experiment.logger) if "logger" in cfg.experiment else None
    if isinstance(logger, WandbLogger):
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
        )

    # ─────────── callbacks ───────────
    callbacks = [instantiate(cb) for cb in cfg.experiment.callbacks.values()]
    ckpt_cb = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)

    # cập nhật chỉ số monitor
    if ckpt_cb:
        ckpt_cb.monitor = "val/macro/f1"  # <── NEW monitor
        ckpt_cb.dirpath = os.path.join(ckpt_cb.dirpath, "run_0")

    # ─────────── trainer ───────────
    trainer = pl.Trainer(
        **cfg.experiment.trainer,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    # ─────────── train ───────────
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # ─────────── lấy checkpoint tốt nhất ───────────
    best_model_path = ckpt_cb.best_model_path
    best_score = ckpt_cb.best_model_score.item()
    pprint(f"Best model path: {best_model_path}")
    pprint(f"Best val f1:     {best_score:.4f}")

    # ─────────── test (log CM & ROC) ───────────

    # ─────────── kết thúc wandb & lưu lên Drive ───────────
    wandb.finish()
    save_and_push_best_model(best_model_path)


if __name__ == "__main__":
    main()
