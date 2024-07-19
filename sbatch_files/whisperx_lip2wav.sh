#!/bin/bash

#SBATCH -A research
#SBATCH --nodelist=gnode075
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 27
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python3 data_processing/whisperx_lip2wav.py --speaker chess --model large-v3