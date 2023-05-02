#!/bin/bash

#SBATCH --job-name=my_job_arturo_try
#SBATCH --time=11:59:00
#SBATCH --mem=40GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/1.2-HyperParameterTuning/logs/1-MasiveHyperParameterTuning_210_epochs_3000_samples_1e-4_lr.out"

source /home/asirvent/pytorch_bien/bin/activate 
python3 /home/asirvent/extra_repos/TFM/1.2-HyperParameterTuning/notebooks/1-MasiveHyperParameterTuning_210_epochs_3000_samples_1e-4_lr.py
