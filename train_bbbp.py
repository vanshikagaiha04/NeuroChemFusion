import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score, f1_score
import warnings

# --- SETUP ---
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")
device = torch.device("cuda")

# --- DATA PREP ---
def prepare_bbbp_v17():
    print(">>> Loading BBBP Data (V17 Slim Mode)...", flush=True)
    featurizer = dc.feat.RDKitDescriptors()
    tasks, datasets, transformers = dc.molnet.load_bbbp(
        featurizer=featurizer, data_dir='./bbbp_data_desc', reload=False
    )
    train_ds, valid_ds, test_ds = datasets

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

# --- ARCHITECTURE (V17: Slim & Robust) ---
class NeuroChemFusionBBBP_V17(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        # Smaller embeddings and Mamba to prevent overfitting
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.mamba_branch = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
        
        # Heavy Dropout (0.7) for a small dataset
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.7),
            nn.Linear(128, 64)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.7),
            nn.Linear(64, 1)
        )

    def forward(self, smiles_ids, descriptors):
        f_seq = self.mamba_branch(self.embedding(smiles_ids)).mean(dim=1)
        f_phys = self.phys_branch(descriptors)
        
        # Concat instead of gating to reduce parameters
        f_fusion = torch.cat([f_seq, f_phys], dim=-1)
        return self.classifier(f_fusion)

# --- UTILS ---
def to_tensors_bbbp(ds, encode_fn):
    s = encode_fn(ds.ids).to(device)
    raw_X = np.nan_to_num(ds.X.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    d = torch.tensor(raw_X, dtype=torch.float32).to(device)
    y = torch.tensor(ds.y, dtype=torch.float32).to(device)
    return s, d, y

def evaluate_bbbp(model, ds, encode_fn):
    model.eval()
    with torch.no_grad():
        s, d, y_true = to_tensors_bbbp(ds, encode_fn)
        logits = model(s, d)
        y_pred = torch.sigmoid(logits).cpu().numpy()
        y_true = y_true.cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int), zero_division=0)
    return auc, f1

# --- EXECUTION ---
train_ds, valid_ds, test_ds, encode_fn, vocab_sz = prepare_bbbp_v17()
t_s, t_d, t_y = to_tensors_bbbp(train_ds, encode_fn)

model = NeuroChemFusionBBBP_V17(vocab_sz, t_d.shape[1]).to(device)

# EXTREME REGULARIZATION: Weight Decay = 0.1
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
criterion = nn.BCEWithLogitsLoss()

print("\n>>> Training V17 (Slim & Robust) on H100...", flush=True)
best_auc = 0.0
patience = 40
patience_counter = 0

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    logits = model(t_s, t_d)
    loss = criterion(logits, t_y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    if epoch % 10 == 0:
        val_auc, _ = evaluate_bbbp(model, valid_ds, encode_fn)
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}", flush=True)
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_bbbp_v17.pt')
        else:
            patience_counter += 10
            
        if patience_counter >= patience:
            print(f">>> Early stopping triggered at Epoch {epoch}", flush=True)
            break

# Final Eval
print("\n>>> Loading Best V17 Model for Test Set...", flush=True)
model.load_state_dict(torch.load('best_bbbp_v17.pt'))
test_auc, test_f1 = evaluate_bbbp(model, test_ds, encode_fn)
print("\n" + "="*50 + f"\nFINAL BBBP TEST ROC-AUC: {test_auc:.4f} | F1: {test_f1:.4f}\n" + "="*50)