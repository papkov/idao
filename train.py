from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from idao.data_module import IDAODataModule
from idao.model import SimpleConv

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)
    root = Path(hydra.utils.get_original_cwd())

    dataset_dm = IDAODataModule(
        data_dir=root/cfg.data.path, batch_size=cfg.train.batch_size, ext=cfg.data.ext
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()

    for mode in ["classification", "regression"]:

        # init model
        model = SimpleConv(mode=mode)
        epochs = (
            cfg.train.epochs_classification
            if mode == "classification"
            else cfg.train.epochs_regression
        )
        # Initialize a trainer
        trainer = pl.Trainer(
            gpus=cfg.train.gpus,
            max_epochs=epochs,
            progress_bar_refresh_rate=20,
            weights_save_path=Path(cfg.train.checkpoint_path).joinpath(mode),
            default_root_dir=Path(cfg.train.checkpoint_path),
        )

        # Train the model ⚡
        trainer.fit(model, dataset_dm)


if __name__ == "__main__":
    main()
