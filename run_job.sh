#!/bin/bash
#PBS -N NeuroChem_General_Job
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -j oe

# 1. Go to your folder
cd /home/kamakshi.rautela/NeuroChem_Minor

# 2. Activate your environment
source ~/.bashrc
conda activate neuro_final

# 3. Optimization flags
export PYTHONUNBUFFERED=1
export FORCE_CUDA=1
export OMP_NUM_THREADS=4

# ==========================================================
# 🛑 CHANGE ONLY THIS LINE TO RUN DIFFERENT PYTHON FILES 🛑
SCRIPT_TO_RUN="generate_graphs.py"
# ==========================================================

echo "=========================================================="
echo "STARTING JOB: $SCRIPT_TO_RUN"
echo "TIME: $(date)"
echo "=========================================================="

# Run the python script
python -u $SCRIPT_TO_RUN

echo "=========================================================="
echo "JOB FINISHED AT: $(date)"
echo "=========================================================="