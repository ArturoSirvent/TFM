#!/bin/bash

#SBATCH --job-name=my_job_arturo_1
#SBATCH --time=11:59:00
#SBATCH --mem=40GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/3-ApplyModel/auto/logs/finetune_pumpData_45_epoch_1e-4_lr_7_warm_3_head_128_dmodel_2_block_128_batch_5_stride.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/3-ApplyModel/auto/scripts/finetune_pumpData_45_epoch_1e-4_lr_7_warm_3_head_128_dmodel_2_block_128_batch_5_stride.py
