#!/bin/bash

#SBATCH -A vanshg
#SBATCH --nodelist=gnode058
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -c 10
#SBATCH --time=96:00:00
#SBATCH --mail-user=vansh.garg@research.iiit.ac.in
#SBATCH --mail-type=ALL

cd  /ssd_scratch/cvit/vanshg

./download_tcd_timid.sh