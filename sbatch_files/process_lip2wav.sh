#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode071
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH -c 36
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd ~/auto_avsr
git checkout personalization

python3 data_processing/process_lip2wav.py --ngpu 4 --speaker dl --detector yolov5 --batch-size 128
