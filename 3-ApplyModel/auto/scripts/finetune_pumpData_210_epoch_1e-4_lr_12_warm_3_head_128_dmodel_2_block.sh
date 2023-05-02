#!/bin/bash

#SBATCH --job-name=my_job_arturo_2
#SBATCH --time=11:59:00
#SBATCH --mem=50GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/3-ApplyModel/auto/logs/finetune_pumpData_210_epoch_1e-4_lr_12_warm_3_head_128_dmodel_2_block.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/3-ApplyModel/auto/scripts/finetune_pumpData_210_epoch_1e-4_lr_12_warm_3_head_128_dmodel_2_block.py
