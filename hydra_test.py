import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.experiment.logger.name)


if __name__ == "__main__":
    main()
