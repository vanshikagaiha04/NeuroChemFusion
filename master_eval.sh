#!/bin/bash
#PBS -N NeuroChem_Final_Eval
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=00:10:00
#PBS -q gpu
#PBS -j oe

# --- Step 1: Directory Navigation ---
cd /home/kamakshi.rautela/NeuroChem_Minor

# --- Step 2: Environment Setup ---
source ~/.bashrc
conda activate neuro_final

# --- Step 3: Flags ---
export PYTHONUNBUFFERED=1
export FORCE_CUDA=1

echo "=========================================================="
echo "LAUNCHING FINAL MASTER EVALUATION"
echo "START TIME: $(date)"
echo "MODELS: BBBP MASTER & TOX21 MASTER"
echo "=========================================================="

# --- Step 4: Execution ---
# 'master_evaluator.py' ko run karega
python -u master_evaluate.py

echo "=========================================================="
echo "EVALUATION FINISHED AT: $(date)"
echo "CHECK THE OUTPUT ABOVE FOR FINAL AUC SCORES"
echo "=========================================================="