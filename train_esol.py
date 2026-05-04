# --- BLOCK 1: Imports & Environment Patches ---
import os
import warnings
# Humne jo C++ crash fix kiya tha, uske liye ye zaroori hai:
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0' 
warnings.filterwarnings("ignore")

import torch
import deepchem as dc
import pandas as pd
from deepchem.models.torch_models import GCNModel

# --- BLOCK 2: Data Loading (Correct Featurizer Object) ---
print("Loading ESOL Dataset with MolGraphConvFeaturizer...")

# Instead of a string, we use the specific Featurizer object needed for GCNModel
dgl_feat = dc.feat.MolGraphConvFeaturizer()

tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=dgl_feat, 
    data_dir='./esol_data', 
    save_dir='./esol_data'
)

train_dataset, valid_dataset, test_dataset = datasets

# --- BLOCK 3: Model Setup (H100 GPU) ---
print(f"Initializing GCNModel on CUDA...")
print(f"Checking GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found'}")

model = GCNModel(
    n_tasks=len(tasks), 
    mode='regression', 
    batch_size=32, 
    learning_rate=0.001,
    device='cuda' # H100 use karne ke liye
)

# --- BLOCK 4: Training ---
print("Starting training on H100 GPU (50 Epochs)...")
model.fit(train_dataset, nb_epoch=50)

# --- BLOCK 5: Evaluation ---
print("Evaluating model performance...")
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("\n" + "="*30)
print(f"TRAINING SUCCESSFUL!")
print(f"Train Pearson R2: {train_scores['pearson_r2_score']:.4f}")
print(f"Test Pearson R2: {test_scores['pearson_r2_score']:.4f}")
print("="*30)