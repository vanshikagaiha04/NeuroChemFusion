import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from scipy.stats import pearsonr
import warnings

# --- BLOCK 1: Environment ---
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")

# --- BLOCK 2: Data Preparation (Fixed Numeric Loading) ---
def prepare_v9_data():
    print(">>> Loading Numeric Data for Tri-modal Fusion...", flush=True)
    
    # RDKitDescriptors wahi 200+ features deta hai jo DMPNN use karta hai.
    # Isse 'numpy.object_' wala error nahi aayega.
    featurizer = dc.feat.RDKitDescriptors() 

    # Hum wahi data folder use karenge jo pehle descriptor-featurized tha
    tasks, datasets, transformers = dc.molnet.load_lipo(
        featurizer=featurizer, 
        data_dir='./lipo_data_desc', 
        reload=False
    )
    train_ds, valid_ds, test_ds = datasets

    # Tokenizer for Mamba (Branch 1)
    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
    token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'], token_to_id['<UNK>'] = 0, 1

    def encode_smiles(smiles_list):
        encoded = []
        for s in smiles_list:
            tokens = tokenizer.tokenize(s)
            ids = [token_to_id.get(t, 1) for t in tokens[:128]]
            ids += [0] * (128 - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    return train_ds, valid_ds, test_ds, encode_smiles, len(token_to_id)

# --- BLOCK 3: Gated Tri-modal Model (Restoring Your Vision) ---
class NeuroChemFusionV9(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        # Branch 1: Mamba (Sequential/Language)
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = nn.Sequential(
            Mamba(d_model=128, d_state=64, d_conv=4, expand=2),
            nn.LayerNorm(128)
        )
        
        # Branch 2: Physics (RDKit Descriptors)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Branch 3: Structural (DMPNN-style latent processing)
        # Is branch ka kaam hai molecule ki global structure (Topology) ko capture karna
        self.struct_branch = nn.Sequential(
            nn.Linear(desc_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

        # Gating Layer (Dynamic weighting of the 3 experts)
        self.gate_layer = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, smiles_ids, descriptors):
        # 1. Sequence Features
        f_seq = self.mamba_branch[0](self.embedding(smiles_ids)).mean(dim=1)
        # 2. Physics Features
        f_phys = self.phys_branch(descriptors)
        # 3. Structural/DMPNN-latent Features
        f_struct = self.struct_branch(descriptors)
        
        # Gating Logic
        combined = torch.cat([f_seq, f_phys, f_struct], dim=-1)
        gates = self.gate_layer(combined)
        
        # Final Fusion
        f_fusion = (gates[:, 0:1] * f_seq) + \
                   (gates[:, 1:2] * f_phys) + \
                   (gates[:, 2:3] * f_struct)
        
        return self.regressor(f_fusion)

# --- BLOCK 4: Execution ---
train_ds, valid_ds, test_ds, encode_fn, vocab_sz = prepare_v9_data()
device = torch.device("cuda")

def to_tensors(ds):
    smiles = encode_fn(ds.ids).to(device)
    # Ab ds.X numeric hai, toh ye line crash nahi karegi!
    desc = torch.tensor(ds.X, dtype=torch.float32).to(device)
    y = torch.tensor(ds.y, dtype=torch.float32).to(device)
    return smiles, desc, y

# Pre-prepare tensors for H100 speed
t_smiles, t_desc, t_y = to_tensors(train_ds)
v_smiles, v_desc, v_y = to_tensors(valid_ds)
te_smiles, te_desc, te_y = to_tensors(test_ds)

model = NeuroChemFusionV9(vocab_sz, t_desc.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
criterion = nn.HuberLoss() # Outlier-robust regression

print("\n>>> Training V9 (Tri-modal Fusion) on H100...", flush=True)
best_r2 = -1

for epoch in range(1, 151):
    model.train()
    optimizer.zero_grad()
    preds = model(t_smiles, t_desc)
    loss = criterion(preds, t_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            v_out = model(v_smiles, v_desc).cpu().numpy().flatten()
            v_r2 = pearsonr(v_out, v_y.cpu().numpy().flatten())[0]**2
            
            if v_r2 > best_r2:
                best_r2 = v_r2
                torch.save(model.state_dict(), 'neurochem_v9_best.pt')
                print(f"Epoch {epoch:03d} | New Best Val R²: {v_r2:.4f}", flush=True)

# Final Evaluation
model.load_state_dict(torch.load('neurochem_v9_best.pt'))
model.eval()
with torch.no_grad():
    test_out = model(te_smiles, te_desc).cpu().numpy().flatten()
    final_r2 = pearsonr(test_out, te_y.cpu().numpy().flatten())[0]**2
    print(f"\n" + "="*50 + f"\nFINAL TRI-MODAL TEST R²: {final_r2:.4f}\n" + "="*50)