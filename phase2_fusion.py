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

class NeuroChemSpecialist(torch.nn.Module):
    def __init__(self, backbone, n_tasks):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(384, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, n_tasks)
        )
    def forward(self, s, d, n): 
        return self.classifier(self.backbone(s, d, n))

# --- NAN-SHIELD BATCHING ---
def get_safe_batch(ds, batch_size=32):
    # Pure dataset se valid indices nikalna
    indices = np.random.choice(len(ds), batch_size)
    X = ds.X[indices].astype(np.float32)
    y = ds.y[indices].astype(np.float32)
    
    # NaN ya Infinity ko zero se replace karna
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    # Agar poori row zero hai (failed featurization), toh noise add karna to prevent collapse
    for i in range(len(X)):
        if not np.any(X[i]): X[i] = np.random.normal(0, 0.01, X.shape[1])
        
    return X, y

def run_specialist(task_type='bbbp'):
    print(f"\n>>> Training {task_type.upper()} with NaN-Shield...", flush=True)
    f_desc = dc.feat.RDKitDescriptors()
    
    loader_func = dc.molnet.load_bbbp if task_type == 'bbbp' else dc.molnet.load_tox21
    d_dir = './bbbp_data_desc' if task_type == 'bbbp' else './tox21_data_desc'
    _, datasets, _ = loader_func(featurizer=f_desc, data_dir=d_dir, reload=False)
    ds = datasets[0] # Train split[cite: 6]

    backbone = SharedBackbone(52, 217).to(device)
    if os.path.exists('neurochem_backbone_weights.pt'):
        backbone.load_state_dict(torch.load('neurochem_backbone_weights.pt', map_location=device))
        print(">>> Backbone Loaded.")
    
    for p in backbone.parameters(): p.requires_grad = False
    
    model = NeuroChemSpecialist(backbone, 1 if task_type == 'bbbp' else 12).to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(1, 101):
        model.train()
        X_b, y_b = get_safe_batch(ds)
        
        # Standard Padding
        if X_b.shape[1] < 217: X_b = np.pad(X_b, ((0,0),(0, 217-X_b.shape[1])))
        else: X_b = X_b[:, :217]
        
        s = torch.randint(0, 52, (32, 128)).to(device)
        d = torch.from_numpy(X_b).to(device)
        n = torch.randn(32, 20).to(device)
        y = torch.from_numpy(y_b).to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(s, d, n), y)
        
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f'neurochem_{task_type}_final.pt')
    print(f">>> Saved: neurochem_{task_type}_final.pt")

if __name__ == '__main__':
    run_specialist('bbbp')
    run_specialist('tox21')