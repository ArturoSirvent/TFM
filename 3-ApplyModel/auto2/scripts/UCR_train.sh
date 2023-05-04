#!/bin/bash

#SBATCH --job-name=my_job_arturo_UCR
#SBATCH --time=10:59:00
#SBATCH --mem=20GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/3-ApplyModel/auto2/logs/UCR_train_1-part4.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/3-ApplyModel/auto2/scripts/UCR_train_1.py
