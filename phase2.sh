#!/bin/bash
#PBS -N NeuroChem_Phase2_Specialist
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -j oe

qstat -n -1
cd /home/kamakshi.rautela/NeuroChem_Minor

source ~/.bashrc
conda activate neuro_final

export PYTHONUNBUFFERED=1
export FORCE_CUDA=1
export OMP_NUM_THREADS=4

echo "=========================================================="
echo "LAUNCHING PHASE 2: TASK TRANSFER (BBBP + TOX21)"
echo "START TIME: $(date)"
echo "NODE: $(hostname)"
echo "GPU INFO: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- Step 4: Execution ---
python -u phase2_fusion.py

echo "=========================================================="
echo "PHASE 2 FINISHED AT: $(date)"
echo "SPECIALIST WEIGHTS SAVED: neurochem_bbbp_final.pt, neurochem_tox21_final.pt"
echo "=========================================================="