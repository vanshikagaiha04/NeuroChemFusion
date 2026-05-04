#!/bin/bash
#PBS -N NeuroChem_BBBP
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=1:00:00
#PBS -q gpu
#PBS -j oe

# --- Step 1: Directory Navigation ---qstat -n -1
cd /home/kamakshi.rautela/NeuroChem_Minor

# --- Step 2: Environment Setup ---
source ~/.bashrc
conda activate neuro_final

# --- Step 3: Performance & Logging Flags ---
# Force Python to print immediately to the .o file
export PYTHONUNBUFFERED=1
# Ensure CUDA is visible to Mamba kernels
export FORCE_CUDA=1

echo "=========================================================="
echo "LAUNCHING BBBP TRI-MODAL FUSION"
echo "START TIME: $(date)"
echo "NODE: $(hostname)"
echo "GPU INFO: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- Step 4: Execution ---
# Using '-u' for unbuffered output as a double safety
python -u train_bbbp_descriptor.py

echo "=========================================================="
echo "JOB FINISHED AT: $(date)"
echo "=========================================================="