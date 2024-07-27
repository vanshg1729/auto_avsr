import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from avg_ckpts import ensemble
from datamodule.data_module_phrase import DataModulePhrase
from lightning_grid import ModelModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from utils.finetune_utils import *
import wandb

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)

    print(f"Inside main() function")
    speaker = cfg.speaker
    print(f"{cfg.finetune = }")

    project_name = "lip2wav_inference"
    run_name = "chem"
    cfg.log_folder = os.path.join(cfg.logging_dir, f"{project_name}/{run_name}")
    cfg.exp_dir = cfg.log_folder
    cfg.trainer.default_root_dir = cfg.log_folder
    os.makedirs(cfg.log_folder, exist_ok=True)
    print(f"\nLogging Directory: {cfg.log_folder}")

    # Logging Stuff
    loggers = []
    csv_logger = CSVLogger(
        save_dir=cfg.log_folder,
        flush_logs_every_n_steps=1
    )
    loggers.append(csv_logger)
    if cfg.wandb:
        wandb_logger = WandbLogger(
            name=run_name,
            project=project_name,
            # config=cfg,
            settings=wandb.Settings(code_dir='.')
        )
        loggers.append(wandb_logger)

    modelmodule = ModelModule(cfg)

    print(f"{cfg.trainer = }")
    datamodule = DataModulePhrase(cfg)
    trainer = Trainer(
        **cfg.trainer,
        strategy='ddp_find_unused_parameters_true',
        logger=loggers,
    )
    print(f"{trainer.num_devices = }")
    print(f"{trainer.device_ids = }")

    trainer.validate(model=modelmodule, verbose=True, datamodule=datamodule)

if __name__ == "__main__":
    main()
