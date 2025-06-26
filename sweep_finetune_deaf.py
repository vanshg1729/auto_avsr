import hydra
from omegaconf import OmegaConf
import os

from finetune_deaf import main

def run_for_finetune_types(finetune_types, base_overrides=None):
    # Set up Hydra to use your config directory
    config_path = "configs"
    config_name = "config"

    with hydra.initialize(config_path=config_path):
        for finetune_type in finetune_types:
            # Compose config with overrides
            overrides = [f"finetune={finetune_type}"]
            if base_overrides:
                overrides += base_overrides
            cfg = hydra.compose(config_name=config_name, overrides=overrides)
            print(f"\n==== Running for finetune_type={finetune_type} ====")
            print(OmegaConf.to_yaml(cfg))
            # calling the finetuning function with this config
            main(cfg)

if __name__ == "__main__":
    finetune_types = ["full", "encoder", "encoders", "encoders_conv_module", "encoders_middle_six"] # different finetuning experiments
    base_overrides = [
        "data.modality=video",
        "data.dataset.root_dir=/ssd_scratch/cvit/akshat/datasets/accented_speakers",
        "data.dataset.root_dir=/ssd_scratch/cvit/akshat/datasets/accented_speakers",
        "data.dataset.train_file=/ssd_scratch/cvit/akshat/datasets/accented_speakers/eleanor/train_reduced_labels.txt",
        "data.dataset.val_file=/ssd_scratch/cvit/akshat/datasets/accented_speakers/eleanor/val_reduced_labels.txt",
        "data.dataset.test_file=/ssd_scratch/cvit/akshat/datasets/accented_speakers/eleanor/test_reduced_labels.txt",
        "pretrained_model_path=/ssd_scratch/cvit/akshat/checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth"
    ]
    run_for_finetune_types(finetune_types, base_overrides)