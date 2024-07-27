#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode076
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH -c 38
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python inference.py data.modality=video \
               data.dataset.root_dir=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset \
               data.dataset.test_file=/ssd_scratch/cvit/vanshg/Lip2Wav/Dataset/chem/labels.txt \
               pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth