#!/bin/bash

#SBATCH --job-name=my_job_arturo_try
#SBATCH --time=11:59:00
#SBATCH --mem=30GB
#SBATCH --partition guest
#SBATCH --output="/home/asirvent/extra_repos/TFM/1-AnomalyTranformer_MIO/ejecucion_remota/logs/execute_2-entrenamiento_PSM.out"

source /home/asirvent/pytorch_env/bin/activate 
cd /home/asirvent/extra_repos/Anomaly-Transformer
bash /home/asirvent/extra_repos/Anomaly-Transformer/scripts/PSM.sh
