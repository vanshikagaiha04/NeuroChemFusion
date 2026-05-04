#!/bin/bash
#PBS -N NeuroChem_Phase1_Pretrain
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -j oe

# --- Step 1: Directory Navigation ---
# qstat se status check karne ke liye command output mein rakha hai
qstat -n -1
cd /home/kamakshi.rautela/NeuroChem_Minor

# --- Step 2: Environment Setup ---
source ~/.bashrc
conda activate neuro_final

# --- Step 3: Performance & Logging Flags ---
export PYTHONUNBUFFERED=1
# Mamba kernels ke liye CUDA flags important hain
export FORCE_CUDA=1
# Threading control for RDKit/DeepChem processing
export OMP_NUM_THREADS=4

echo "=========================================================="
echo "LAUNCHING PHASE 1: BACKBONE PRE-TRAINING (ESOL + LIPO)"
echo "START TIME: $(date)"
echo "NODE: $(hostname)"
echo "GPU INFO: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- Step 4: Execution ---
# 'pretrain_neurochem.py' wo file hai jo maine pichle response mein di thi
python -u phase1_fusion.py

echo "=========================================================="
echo "PHASE 1 FINISHED AT: $(date)"
echo "BACKBONE WEIGHTS SHOULD BE SAVED IN: neurochem_backbone_weights.pt"
echo "=========================================================="