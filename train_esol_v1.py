import os
import warnings

# Humne jo C++ crash fix kiya tha, uske liye ye zaroori hai:
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0' 
warnings.filterwarnings("ignore")

import torch
import deepchem as dc
from deepchem.models.torch_models import GCNModel

# --- BLOCK 1: Data Loading & Enhanced Featurizer ---
print("Loading ESOL Dataset with Enhanced MolGraphConvFeaturizer...")

# Include bond features and chirality for richer molecular representations
dgl_feat = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)

tasks, datasets, transformers = dc.molnet.load_delaney(
    featurizer=dgl_feat, 
    data_dir='./esol_data', 
    save_dir='./esol_data'
)

train_dataset, valid_dataset, test_dataset = datasets

# --- BLOCK 2: Model Setup with Regularization (H100 GPU) ---
print(f"Initializing Tuned GCNModel on CUDA...")
print(f"Checking GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU found'}")

model = GCNModel(
    n_tasks=len(tasks), 
    mode='regression', 
    number_atom_features=32,      # <--- ADD THIS LINE (30 + 2 chirality features = 32)
    batch_size=64,                
    learning_rate=0.0005,         
    graph_conv_layers=[128, 128], 
    dropout=0.1,                  
    predictor_dropout=0.1,        
    device='cuda',                
    model_dir='./best_gcn_model'  
)

# --- BLOCK 3: Validation Callback & Training ---
print("Starting training on H100 GPU with Validation Checkpoint...")
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

# Explicitly name transformers=transformers so it doesn't overwrite output_file
callback = dc.models.ValidationCallback(
        valid_dataset, 
        50,                           # Check every 50 steps
        [metric], 
        transformers=transformers,    # <--- FIXED LINE
        save_dir='./best_gcn_model', 
        save_on_minimum=False         # False because we want to MAXIMIZE R2 score
)

# Train for more epochs (100), but rely on the callback to save the optimal weights
model.fit(train_dataset, nb_epoch=100, callbacks=[callback])

# --- BLOCK 4: Restore Best Weights & Evaluation ---
print("Restoring best model weights and evaluating performance...")
# Restore the best weights found during training (highest validation score)
model.restore()

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)
test_scores = model.evaluate(test_dataset, [metric], transformers)

print("\n" + "="*35)
print(f"TRAINING & EVALUATION SUCCESSFUL!")
print(f"Train Pearson R2: {train_scores['pearson_r2_score']:.4f}")
print(f"Valid Pearson R2: {valid_scores['pearson_r2_score']:.4f}")
print(f"Test Pearson R2:  {test_scores['pearson_r2_score']:.4f}")
print("="*35)