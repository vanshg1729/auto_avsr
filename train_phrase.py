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
import wandb

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.trainer.devices = torch.cuda.device_count()

    speaker = "vansh"
    train_size = cfg.train_size

    project_name = "auto_avsr_phrase_finetuning"
    run_name = f"{speaker}_{train_size}_full_finetune_default"
    cfg.log_folder = os.path.join(cfg.logging_dir, f"{project_name}/{run_name}")
    cfg.exp_dir = cfg.log_folder
    cfg.trainer.default_root_dir = cfg.log_folder
    os.makedirs(cfg.log_folder, exist_ok=True)
    print(f"\nLogging Directory: {cfg.log_folder}")

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        verbose=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        # dirpath=cfg.log_folder,
        save_last=True,
        # filename="{epoch}",
        save_top_k=-1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Logging Stuff
    loggers = []
    logger = CSVLogger(
        save_dir=cfg.log_folder,
        flush_logs_every_n_steps=1
    )
    loggers.append(logger)

    if cfg.wandb:
        wandb_logger = WandbLogger(
            name=run_name,
            project=project_name,
            # config=cfg,
            settings=wandb.Settings(code_dir='.')
        )
        loggers.append(wandb_logger)

    modelmodule = ModelModule(cfg)
    datamodule = DataModulePhrase(cfg)
    # train_dataloader = datamodule.train_dataloader()
    trainer = Trainer(
        **cfg.trainer,
        # logger=loggers,
        #logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),
        # callbacks=callbacks,
    )

    trainer.fit(model=modelmodule, datamodule=datamodule)
    # ensemble(cfg)

if __name__ == "__main__":
    main()
