# --- BLOCK 1: Imports & Patches ---
import os
import warnings
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0' 
warnings.filterwarnings("ignore")

import torch
import deepchem as dc
from deepchem.models.torch_models import DMPNNModel

# --- BLOCK 2: Data Loading ---
print("Loading ESOL Dataset...")
dmpnn_feat = dc.feat.DMPNNFeaturizer(features_generators=['rdkit_2d_normalized'])

tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=dmpnn_feat, 
    data_dir='./esol_dmpnn', 
    save_dir='./esol_dmpnn'
)
train_dataset, valid_dataset, test_dataset = datasets

# --- BLOCK 3: Regularized Model Setup ---
print(f"Initializing Regularized DMPNNModel on CUDA...")

model = DMPNNModel(
    n_tasks=len(tasks),
    batch_size=32,
    learning_rate=0.0005,  # Lowered LR for better generalization
    dropout=0.3,           # Adding 30% Dropout to kill overfitting
    weight_decay=1e-4,     # L2 Regularization
    mode='regression',
    device='cuda'
)

# --- BLOCK 4: Training with Validation Tracking ---
num_epochs = 60 # Thode extra epochs kyunki LR kam kiya hai
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

print(f"Starting Regularized Training...")
print("-" * 50)

for epoch in range(1, num_epochs + 1):
    loss = model.fit(train_dataset, nb_epoch=1)
    
    # Har 10 epochs par progress check karenge
    if epoch % 10 == 0:
        val_score = model.evaluate(valid_dataset, [metric], transformers)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val R2: {val_score['pearson_r2_score']:.4f}")

print("-" * 50)

# --- BLOCK 5: Final Evaluation ---
train_scores = model.evaluate(train_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("\n" + "="*30)
print(f"REGULARIZED D-MPNN RESULTS")
print(f"Train Pearson R2: {train_scores['pearson_r2_score']:.4f}")
print(f"Test Pearson R2: {test_scores['pearson_r2_score']:.4f}")
print("="*30)