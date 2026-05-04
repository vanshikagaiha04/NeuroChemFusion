import os, torch, numpy as np, deepchem as dc
from mamba_ssm import Mamba
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STANDARDIZED ARCHITECTURE ---
class SharedBackbone(torch.nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = Mamba(d_model=128, d_state=64, d_conv=4, expand=2)
        self.phys_branch = torch.nn.Sequential(
            torch.nn.Linear(desc_dim, 256), torch.nn.LayerNorm(256), torch.nn.GELU(),
            torch.nn.Linear(256, 128), torch.nn.LayerNorm(128)
        )
        self.graph_branch = torch.nn.Sequential(torch.nn.Linear(20, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128))

    def forward(self, s, d, n):
        f_seq = self.mamba_branch(self.embedding(s)).mean(dim=1)
        f_phys = self.phys_branch(d)
        f_graph = self.graph_branch(n).mean(dim=0, keepdim=True).repeat(f_seq.size(0), 1)
        return torch.cat([f_seq, f_phys, f_graph], dim=-1)

class NeuroChemFinal(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Specialist Heads for Multi-task
        self.bbbp_head = torch.nn.Linear(384, 1)
        self.tox_head = torch.nn.Linear(384, 12)
        self.esol_head = torch.nn.Linear(384, 1)
        self.lipo_head = torch.nn.Linear(384, 1)

    def forward(self, s, d, n, task='bbbp'):
        latent = self.backbone(s, d, n)
        if task == 'bbbp': return self.bbbp_head(latent)
        if task == 'tox21': return self.tox_head(latent)
        if task == 'esol': return self.esol_head(latent)
        return self.lipo_head(latent)

def train_master():
    f_desc = dc.feat.RDKitDescriptors()
    print(">>> Phase 3: Loading All Datasets for Global Sync...", flush=True)
    all_ds = {
        'bbbp': dc.molnet.load_bbbp(featurizer=f_desc, data_dir='./bbbp_data_desc', reload=False)[1][0],
        'tox21': dc.molnet.load_tox21(featurizer=f_desc, data_dir='./tox21_data_desc', reload=False)[1][0],
        'esol': dc.molnet.load_delaney(featurizer=f_desc, data_dir='./esol_data', reload=False)[1][0],
        'lipo': dc.molnet.load_lipo(featurizer=f_desc, data_dir='./lipo_data', reload=False)[1][0]
    }

    backbone = SharedBackbone(52, 217).to(device)
    model = NeuroChemFinal(backbone).to(device)

    # --- INJECT PHASE 2 WEIGHTS ---
    if os.path.exists('neurochem_bbbp_final.pt'):
        sd = torch.load('neurochem_bbbp_final.pt', map_location=device)
        # Load backbone and bbbp head from Phase 2
        model.load_state_dict(sd, strict=False)
        print(">>> SUCCESS: Specialist Knowledge Injected.")

    # UNFREEZE ALL for Global Fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    print(">>> Training Master Fusion (Global Sync)...", flush=True)
    for epoch in range(1, 101):
        model.train()
        for task in ['bbbp', 'tox21', 'esol', 'lipo']:
            ds = all_ds[task]
            idx = np.random.choice(len(ds), 32)
            X_b = np.nan_to_num(ds.X[idx].astype(np.float32))
            if X_b.shape[1] < 217: X_b = np.pad(X_b, ((0,0),(0, 217-X_b.shape[1])))
            else: X_b = X_b[:, :217]

            s, d, n, y = torch.randint(0, 52, (32, 128)).to(device), torch.from_numpy(X_b).to(device), torch.randn(32, 20).to(device), torch.from_numpy(np.nan_to_num(ds.y[idx])).float().to(device)
            
            optimizer.zero_grad()
            out = model(s, d, n, task=task)
            loss = torch.nn.BCEWithLogitsLoss()(out, y) if task in ['bbbp', 'tox21'] else torch.nn.HuberLoss()(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if epoch % 25 == 0: print(f"Epoch {epoch} | Syncing...")

    torch.save(model.state_dict(), 'neurochem_UNIVERSAL_MASTER.pt')
    print(">>> UNIVERSAL MASTER SAVED.")

if __name__ == '__main__': train_master()