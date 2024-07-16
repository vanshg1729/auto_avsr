#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode056
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 10
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python demo.py  data.modality=video \
                pretrained_model_path=./checkpoints/lrs3/models/LRS3_V_WER19.1/model.pth \
                file_path=/ssd_scratch/cvit/vanshg/vansh_phrases/processed_videos/1719055516858_37_6676b49c9cb47e5d1fc7a569.mp4