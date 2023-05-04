#!/bin/bash

#SBATCH --job-name=my_job_arturo_UCR
#SBATCH --time=11:59:00
#SBATCH --mem=20GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/3-ApplyModel/auto2/logs/UCR_train_1.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/3-ApplyModel/auto2/scripts/UCR_train_1.py
