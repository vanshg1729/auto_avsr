import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from avg_ckpts import ensemble
from datamodule.data_module_phrase import DataModulePhrase
from lightning_phrase import ModelModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from utils.finetune_utils import *
import wandb

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)

    print(f"Inside main() function")

    speaker = "dl"
    project_name = f"{speaker}_inference"
    # run_name = "chem_full_finetuning_const_lr0.0001_wd0.5"
    run_name = f"train_pretrained"
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

    # Loading the Model
    modelmodule = ModelModule(cfg)

    print(f"{cfg.trainer = }")
    datamodule = DataModulePhrase(cfg)
    trainer = Trainer(
        **cfg.trainer,
        logger=loggers,
    )
    print(f"{trainer.num_devices = }")
    print(f"{trainer.device_ids = }")

    # ckpt_path = "/ssd_scratch/cvit/vanshg/auto_avsr_chem_finetune/chem_full_finetuning_const_lr0.0001_wd0.5/lightning_logs/version_0/checkpoints/epoch=0-step=1616.ckpt"
    trainer.validate(model=modelmodule, verbose=True, datamodule=datamodule)

if __name__ == "__main__":
    main()
