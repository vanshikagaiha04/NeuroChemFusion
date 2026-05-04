#!/bin/bash
#PBS -N NeuroChem_V10
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=10:00:00
#PBS -q gpu
#PBS -j oe

cd /home/kamakshi.rautela/NeuroChem_Minor
source ~/.bashrc
conda activate neuro_final

# Force unbuffered output for live tracking
export PYTHONUNBUFFERED=1

echo "Starting NeuroChem Fusion V16 (SOTA Cross-Modal) Run on H100..."
python neurochem_fusion_v16.py