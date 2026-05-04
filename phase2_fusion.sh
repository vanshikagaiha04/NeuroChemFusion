#!/bin/bash
#PBS -N NeuroChem_Phase2
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=10:00:00
#PBS -q gpu
#PBS -j oe

cd /home/kamakshi.rautela/NeuroChem_Minor

source ~/.bashrc
conda activate neuro_final

export PYTHONUNBUFFERED=1
export FORCE_CUDA=1

echo "=========================================================="
echo "LAUNCHING PHASE 2: TRANSFER LEARNING"
echo "CHECKING BACKBONE: $(ls -lh neurochem_backbone_weights.pt)"
echo "=========================================================="

# Run the Phase 2 script
python -u phase2_fusion.py

echo "=========================================================="
echo "PHASE 2 COMPLETED AT: $(date)"
echo "=========================================================="