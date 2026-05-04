# --- BLOCK 1: Imports & Patches ---
import os
import warnings
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0' 
warnings.filterwarnings("ignore")

import torch
import deepchem as dc
from deepchem.models.torch_models import DMPNNModel

# --- BLOCK 2: Data Loading ---
print("Loading Lipophilicity Dataset...")
print("(Using pre-featurized local cache - no internet needed)")
dmpnn_feat = dc.feat.DMPNNFeaturizer(features_generators=['rdkit_2d_normalized'])

tasks, datasets, transformers = dc.molnet.load_lipo(
    featurizer=dmpnn_feat, 
    data_dir='./lipo_dmpnn', 
    reload=False
)
train_dataset, valid_dataset, test_dataset = datasets

# --- BLOCK 3: Regularized Model Setup ---
print(f"Initializing Regularized DMPNNModel on CUDA...")

model = DMPNNModel(
    n_tasks=len(tasks),
    batch_size=32,
    learning_rate=0.0005,  # Lowered LR for better generalization
    dropout=0.3,           # Adding 30% Dropout to reduce overfitting
    weight_decay=1e-4,     # L2 Regularization
    mode='regression',
    device='cuda'
)

# --- BLOCK 4: Training with Validation Tracking ---
num_epochs = 60
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

print(f"Starting Regularized Training...")
print("-" * 50)

for epoch in range(1, num_epochs + 1):
    loss = model.fit(train_dataset, nb_epoch=1)
    
    # Check progress every 10 epochs
    if epoch % 10 == 0:
        val_score = model.evaluate(valid_dataset, [metric], transformers)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val R2: {val_score['pearson_r2_score']:.4f}")

print("-" * 50)

# --- BLOCK 5: Final Evaluation ---
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("\n" + "="*40)
print(f"REGULARIZED D-MPNN - LIPOPHILICITY RESULTS")
print(f"Train Pearson R²: {train_scores['pearson_r2_score']:.4f}")
print(f"Valid Pearson R²: {valid_scores['pearson_r2_score']:.4f}")
print(f"Test Pearson R²: {test_scores['pearson_r2_score']:.4f}")
print("="*40)
