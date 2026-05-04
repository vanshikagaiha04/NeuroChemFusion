#!/bin/bash
#PBS -N Tox21_Final_Eval
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=02:00:00
#PBS -q gpu
#PBS -j oe

cd /home/kamakshi.rautela/NeuroChem_Minor
source ~/.bashrc
conda activate neuro_final

echo "Running Comprehensive Evaluation for Tox21..."
python evaluate_tox21.py