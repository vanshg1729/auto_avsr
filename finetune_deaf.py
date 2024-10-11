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
    speaker = cfg.speaker
    speaker = "jack"
    finetune_type = cfg.finetune
    print(f"{cfg.finetune = }")
    assert finetune_type in finetune_funcs.keys(), f"{finetune_type} not available"
    finetune_func = finetune_funcs[finetune_type]

    # Parameters
    lr = cfg.optimizer.lr
    wd = cfg.optimizer.weight_decay
    ds_args = cfg.data.dataset
    window = ds_args.time_mask_window
    stride = ds_args.time_mask_stride
    dropout = cfg.model.visual_backbone.dropout_rate
    beam_size = cfg.decode.beam_size

    # Project Name and Folders
    project_name = f"auto_avsr_{speaker}_finetuning"
    # run_name = f"{speaker}_{finetune_type}_finetuning_const_lr{lr}_wd{wd}"
    run_name = f"{speaker}_{finetune_type}_finetuning_const_step_lr{lr}_wd{wd}_win{window}_stride{stride}_drop{dropout}_beam{beam_size}"
    # run_name = f"{speaker}_{finetune_type}_finetuning_train2400_40_const_step_lr{lr}_wd{wd}_win{window}_stride{stride}_drop{dropout}_beam{beam_size}"
    # run_name = f"{speaker}_freeze_frontend3D_finetuning_default_lr1e-4"
    cfg.log_folder = os.path.join(cfg.logging_dir, f"{project_name}/{run_name}")
    cfg.exp_dir = cfg.log_folder
    cfg.trainer.default_root_dir = cfg.log_folder
    os.makedirs(cfg.log_folder, exist_ok=True)
    print(f"\nLogging Directory: {cfg.log_folder}")

    # LR and Checkpoint callbacks
    checkpoint = ModelCheckpoint(
        # monitor="monitoring_step",
        # mode="max",
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

    # Creating the Model Object
    modelmodule = ModelModule(cfg)
    finetune_func(modelmodule.model)
    trainable_params = 0
    for name, param in modelmodule.model.named_parameters():
        if param.requires_grad == True:
            print(f"{name} | {param.shape = } | {param.requires_grad = }")
            trainable_params += param.numel()
    print(f"Total trainable params: {trainable_params}")
    # freeze_frontend3D(modelmodule.model)

    # Creating the Trainer Object
    print(f"{cfg.trainer = }")
    datamodule = DataModulePhrase(cfg)
    trainer = Trainer(
        **cfg.trainer,
        strategy='ddp_find_unused_parameters_true',
        logger=loggers,
        callbacks=callbacks,
    )
    print(f"{trainer.num_devices = }")
    print(f"{trainer.device_ids = }")

    # ckpt_path = "/ssd_scratch/cvit/vanshg/auto_avsr_benny_finetuning/benny_encoders_bottom_half_finetuning_const_lr0.0001_wd1.0_win15_stride25_drop0.1_beam10/lightning_logs/version_0/checkpoints/epoch=10-step=1287.ckpt"
    trainer.fit(model=modelmodule, datamodule=datamodule)
    # trainer.validate(model=modelmodule, verbose=True, datamodule=datamodule)
    # ensemble(cfg)

if __name__ == "__main__":
    main()
