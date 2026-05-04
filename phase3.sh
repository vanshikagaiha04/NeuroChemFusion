#!/bin/bash
#PBS -N NeuroChem_Phase3_Finetune
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
echo "LAUNCHING PHASE 3: GLOBAL FINE-TUNING (SYNCING ALL)"
echo "START TIME: $(date)"
echo "NODE: $(hostname)"
echo "GPU INFO: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================================="

# --- Step 4: Execution ---
python -u phase3_fusion.py

echo "=========================================================="
echo "PHASE 3 FINISHED AT: $(date)"
echo "MASTER WEIGHTS SAVED: neurochem_bbbp_MASTER.pt, neurochem_tox21_MASTER.pt"
echo "=========================================================="