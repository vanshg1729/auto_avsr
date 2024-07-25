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

python3 data_processing/preprocess_test.py