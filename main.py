import os
from pprint import pprint

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from src.utils.save_ckpt import save_and_push_best_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:

    if cfg.user is None:
        raise hydra.errors.HydraException(
            "You must set the 'user' field in the config to your Wandb username."
            " Use +user=<your_username> to set it."
        )

    pprint(cfg)
    best_models = []
    best_scores = []

    for fold in range(cfg.experiment.datamodule.num_folds):
        cfg.experiment.datamodule.current_fold = fold

        model = instantiate(cfg.experiment.model)
        data = instantiate(cfg.experiment.datamodule)

        logger = (
            instantiate(cfg.experiment.logger) if "logger" in cfg.experiment else None
        )

        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(
                OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
            )

        # Instantiate callbacks từ config
        callbacks = [instantiate(cb) for cb in cfg.experiment.callbacks.values()]

        # Tìm checkpoint callback trong list callback
        checkpoint_callback = next(
            (cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None
        )

        # Nếu có checkpoint callback thì chỉnh đường dẫn và metric monitor
        if checkpoint_callback:
            checkpoint_callback.dirpath = os.path.join(
                checkpoint_callback.dirpath, f"fold_{fold}"
            )
            checkpoint_callback.monitor = f"val/fold_{fold}/macro/f1"

            trainer = pl.Trainer(
                **cfg.experiment.trainer,
                logger=logger,
                callbacks=[checkpoint_callback],
                enable_progress_bar=True,
            )

            # Train the model
            trainer.fit(model, data)

            best_models.append(trainer.checkpoint_callback.best_model_path)
            best_scores.append(trainer.checkpoint_callback.best_model_score.item())

        break

    best_model = best_models[best_scores.index(max(best_scores))]

    pprint(f"Best model path: {best_model}")
    # save_and_push_best_model(best_model_path=best_model)


if __name__ == "__main__":
    main()
