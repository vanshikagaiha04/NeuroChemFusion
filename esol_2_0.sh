#!/bin/bash
#PBS -N NeuroChem_ESOL_v17
#PBS -q gpu
#PBS -l nodes=1:ppn=64
#PBS -l mem=80gb
#PBS -l walltime=01:00:00
#PBS -o esol_v17_output.log
#PBS -e esol_v17_error.log

# 1. Move to your project directory
cd /home/kamakshi.rautela/NeuroChem_Minor/

# 2. Activate the correct conda environment
source /home/kamakshi.rautela/miniconda3/bin/activate neuro_final

# 3. Environment Variables for Stability
export TF_CPP_MIN_LOG_LEVEL=3
export DGL_ENABLE_GRAPHBOLT=0
# Disable oneDNN custom ops to prevent numerical jitter
export TF_ENABLE_ONEDNN_OPTS=0

# 4. Print GPU details for the log
echo "=========================================================="
echo "LAUNCHING ESOL V17: TRI-MODAL PEAK PERFORMANCE"
echo "START TIME: $(date)"
echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="

# 5. Execute the Python script
# -u flag is used for unbuffered output to see logs in real-time
python -u esol_v17.py

echo "=========================================================="
echo "ESOL V17 FINISHED AT: $(date)"
echo "CHECK OUTPUT GRAPHS: esol_v17_training_metrics.png, esol_v17_parity_plot.png"
echo "=========================================================="