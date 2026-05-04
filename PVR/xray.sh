#!/bin/bash
#PBS -N ChestXray_ViT
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=10:00:00
#PBS -q gpu
#PBS -j oe

cd /home/kamakshi.rautela/NeuroChem_Minor/PVR
source ~/.bashrc
conda activate neuro_final

export PYTHONUNBUFFERED=1

echo "Starting ViT ChestXray training..."
python train_vit_xray.py