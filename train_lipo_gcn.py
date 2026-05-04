# --- BLOCK 1: Imports & Environment Patches ---
import os
import warnings
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0' 
warnings.filterwarnings("ignore")

import torch
import deepchem as dc
from deepchem.models.torch_models import GCNModel

# --- BLOCK 2: Data Loading (GCN) ---
print("Loading Lipophilicity Dataset with MolGraphConvFeaturizer...")
print("(Using pre-featurized local cache - no internet needed)")

# Use specific Featurizer object for GCNModel
dgl_feat = dc.feat.MolGraphConvFeaturizer()

tasks, datasets, transformers = dc.molnet.load_lipo(
    featurizer=dgl_feat, 
    data_dir='./lipo_data', 
    reload=False
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
    device='cuda'
)

# --- BLOCK 4: Training ---
print("Starting training on H100 GPU (50 Epochs)...")
model.fit(train_dataset, nb_epoch=50)

# --- BLOCK 5: Evaluation ---
print("Evaluating model performance...")
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("\n" + "="*40)
print(f"GCN MODEL - LIPOPHILICITY RESULTS")
print(f"Train Pearson R²: {train_scores['pearson_r2_score']:.4f}")
print(f"Valid Pearson R²: {valid_scores['pearson_r2_score']:.4f}")
print(f"Test Pearson R²: {test_scores['pearson_r2_score']:.4f}")
print("="*40)
