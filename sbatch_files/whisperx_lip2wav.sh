#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode073
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 18
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python3 data_processing/whisperx_lip2wav.py --speaker hs --model large-v3