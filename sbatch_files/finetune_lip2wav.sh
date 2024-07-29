#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode070
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH -c 38
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python finetune_lip2wav.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset \
               data.dataset.train_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/train_labels.txt \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/test_labels.txt \
               data.dataset.val_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/val_labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth