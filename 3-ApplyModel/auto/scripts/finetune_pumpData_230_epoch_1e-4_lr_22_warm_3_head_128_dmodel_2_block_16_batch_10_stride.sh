#!/bin/bash

#SBATCH --job-name=my_job_arturo_2
#SBATCH --time=11:59:00
#SBATCH --mem=40GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/3-ApplyModel/auto/logs/finetune_pumpData_230_epoch_1e-4_lr_22_warm_3_head_128_dmodel_2_block_16_batch_10_stride.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/3-ApplyModel/auto/scripts/finetune_pumpData_230_epoch_1e-4_lr_22_warm_3_head_128_dmodel_2_block_16_batch_10_stride.py
