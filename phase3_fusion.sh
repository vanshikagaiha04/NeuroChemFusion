#!/bin/bash
#PBS -N NeuroChem_Phase3
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q gpu
#PBS -j oe

cd /home/kamakshi.rautela/NeuroChem_Minor
source ~/.bashrc
conda activate neuro_final

export PYTHONUNBUFFERED=1

echo "=========================================================="
echo "LAUNCHING PHASE 3: GLOBAL MASTER SYNC (UNFROZEN)"
echo "START TIME: $(date)"
echo "=========================================================="

python -u phase3_fusion.py

echo "=========================================================="
echo "PHASE 3 COMPLETE. MASTER MODELS GENERATED."
echo "=========================================================="