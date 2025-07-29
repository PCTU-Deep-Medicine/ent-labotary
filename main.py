from pprint import pprint
from timm import create_model
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    pprint(cfg)

    model = instantiate(cfg.experiment.model)
    data = instantiate(cfg.experiment.datamodule)
    logger = instantiate(cfg.experiment.logger) if "logger" in cfg.experiment else None

    # Log config to wandb
    if isinstance(logger, WandbLogger):
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Instantiate callbacks
    callbacks = [instantiate(cfg.experiment.callbacks[cb_name]) for cb_name in cfg.experiment.callbacks] if "callbacks" in cfg.experiment else []
    checkpoint_callback = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)

    # Initialize Trainer
    trainer = pl.Trainer(
        **cfg.experiment.trainer,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, data)

    # Log best model score to wandb
    if isinstance(logger, WandbLogger) and checkpoint_callback is not None:
        best_score = checkpoint_callback.best_model_score.item() if checkpoint_callback.best_model_score else None
        best_path = checkpoint_callback.best_model_path

        logger.experiment.log({
            "best_model_score": best_score,
            "best_model_path": best_path
        })

if __name__ == "__main__":
    main()