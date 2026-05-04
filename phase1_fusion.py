import os, torch, warnings
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBAL STANDARDIZED BACKBONE ---
class SharedBackbone(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = Mamba(d_model=128, d_state=64, d_conv=4, expand=2)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        # Standardized 20-dim Graph input
        self.graph_branch = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, s, d, n):
        f_seq = self.mamba_branch(self.embedding(s)).mean(dim=1)
        f_phys = self.phys_branch(d)
        f_graph = self.graph_branch(n).mean(dim=0, keepdim=True).repeat(f_seq.size(0), 1)
        return torch.cat([f_seq, f_phys, f_graph], dim=-1)

class PretrainHeads(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.esol_head = nn.Linear(384, 1) # 128*3 = 384
        self.lipo_head = nn.Linear(384, 1)

    def forward(self, s, d, n, task='esol'):
        latent = self.backbone(s, d, n)
        return self.esol_head(latent) if task == 'esol' else self.lipo_head(latent)

def train_backbone():
    f_desc = dc.feat.RDKitDescriptors()
    # Explicitly load data from local dirs to avoid network errors
    _, esol_ds, _ = dc.molnet.load_delaney(featurizer=f_desc, data_dir='./esol_data', reload=False)
    _, lipo_ds, _ = dc.molnet.load_lipo(featurizer=f_desc, data_dir='./lipo_data', reload=False)
    
    # Standard Tokenizer
    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
    token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'], token_to_id['<UNK>'] = 0, 1

    def encode(smi):
        tokens = tokenizer.tokenize(smi)
        ids = [token_to_id.get(t, 1) for t in tokens[:128]]
        return ids + [0]*(128-len(ids))

    # Initialize with 217 descriptors (standard RDKit)
    model = PretrainHeads(SharedBackbone(len(token_to_id), 217)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(">>> Phase 1: Backbone Pre-training Started...")
    for epoch in range(1, 51):
        for ds, task in [(esol_ds[0], 'esol'), (lipo_ds[0], 'lipo')]:
            idx = np.random.choice(len(ds), 32)
            # Ensure X has 217 dims
            X_batch = np.nan_to_num(ds.X[idx].astype(np.float32))
            if X_batch.shape[1] < 217: X_batch = np.pad(X_batch, ((0,0),(0, 217-X_batch.shape[1])))
            
            s = torch.tensor([encode(i) for i in ds.ids[idx]]).to(device)
            d = torch.from_numpy(X_batch).to(device)
            n = torch.randn(32, 20).to(device)
            y = torch.tensor(ds.y[idx]).float().to(device)
            
            optimizer.zero_grad()
            loss = nn.HuberLoss()(model(s, d, n, task=task), y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0: print(f"Epoch {epoch} complete.")

    torch.save(model.backbone.state_dict(), 'neurochem_backbone_weights.pt')
    print(">>> Phase 1 Weights Saved.")

if __name__ == '__main__': train_backbone()