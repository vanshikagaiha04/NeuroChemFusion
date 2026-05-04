import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba

# --- BLOCK 1: Environment Patches ---
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'

# --- BLOCK 2: Correct Tri-modal Data Prep ---
def prepare_v4_data():
    print("Loading Data with RDKit Descriptors (Physics Branch)...")
    
    # RDKitDescriptors numeric array deta hai (200+ features)
    # Isse 'TypeError: numpy.object_' wala masla solve ho jayega
    featurizer = dc.feat.RDKitDescriptors() 
    
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer=featurizer, data_dir='./esol_v4'
    )
    
    # Bemis-Murcko Scaffold Splitting for V4 Research Grade
    splitter = dc.splits.ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(datasets[0])
    
    # Tokenizer Mapping (Same as before)
    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's']
    token_to_id = {t: i+1 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'] = 0

    def encode_smiles(smiles_list):
        encoded = []
        for s in smiles_list:
            tokens = tokenizer.tokenize(s)
            ids = [token_to_id[t] for t in tokens[:64] if t in token_to_id]
            ids += [0] * (64 - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded)

    return train_ds, valid_ds, test_ds, encode_smiles, transformers

# --- BLOCK 3: Gated Tri-modal Architecture ---
class NeuroChemFusionV4(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        # Branch 1: Mamba (Sequential Semantics)
        self.embedding = nn.Embedding(vocab_size, 128)
        self.mamba_branch = Mamba(d_model=128, d_state=16, d_conv=4, expand=2)
        
        # Branch 2: Physics Branch (Theoretical Descriptors)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Branch 3: Structural Latent Branch
        self.struct_branch = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.Tanh() # Structural features typically use Tanh for normalization
        )

        # Sigmoid Gating Layer (The V4 Signature)
        self.gate_layer = nn.Sequential(
            nn.Linear(128 * 3, 3),
            nn.Sigmoid()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, smiles_ids, descriptors):
        # Feature Extraction
        f_seq = self.mamba_branch(self.embedding(smiles_ids))[:, -1, :]
        f_phys = self.phys_branch(descriptors)
        f_struct = self.struct_branch(descriptors)
        
        # Dynamic Gating
        combined = torch.cat([f_seq, f_phys, f_struct], dim=-1)
        gates = self.gate_layer(combined)
        
        # Weighted Fusion
        f_fusion = (gates[:, 0:1] * f_seq) + (gates[:, 1:2] * f_phys) + (gates[:, 2:3] * f_struct)
        return self.regressor(f_fusion)

# --- BLOCK 4: Execution ---
train_ds, valid_ds, test_ds, encode_fn, transformers = prepare_v4_data()

# Ab train_ds.X numeric hai, toh .cuda() kaam karega!
train_smiles = encode_fn(train_ds.ids).cuda()
train_desc = torch.tensor(train_ds.X).float().cuda()
train_y = torch.tensor(train_ds.y).float().cuda()

model = NeuroChemFusionV4(vocab_size=30, desc_dim=train_desc.shape[1]).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

print("\nStarting NeuroChem Fusion V4 Training (Gated Tri-modal)...")
for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    
    preds = model(train_smiles, train_desc)
    loss = criterion(preds, train_y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}/100 | Loss: {loss.item():.4f}")

print("\nSUCCESS: V4 Hybrid model trained successfully on H100!")