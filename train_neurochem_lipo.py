import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from scipy.stats import pearsonr
import warnings
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")

MAX_LEN = 128

# ==============================================================================
# 1. Custom Graph Featurization (Node & Edge Features)
# ==============================================================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1, 10)), np.zeros((2, 0), dtype=np.int64)

    atom_features = []
    for atom in mol.GetAtoms():
        features = [atom.GetAtomicNum(), atom.GetDegree(), atom.GetTotalNumHs(), int(atom.GetIsAromatic())]
        features += [0] * (10 - len(features))
        atom_features.append(features)
    
    node_features = np.array(atom_features, dtype=np.float32)

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        
    if len(edges) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array(edges, dtype=np.int64).T

    return node_features, edge_index

# ==============================================================================
# 2. Data Preparation
# ==============================================================================
def prepare_data():
    print(">>> Loading Data & RDKit Descriptors...", flush=True)
    data_path = './lipo_data_desc'

    featurizer = dc.feat.RDKitDescriptors()
    tasks, datasets, _ = dc.molnet.load_lipo(featurizer=featurizer, data_dir=data_path, reload=False)

    splitter = dc.splits.ScaffoldSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(datasets[0])

    print(">>> Applying Normalization...", flush=True)
    norm_X = dc.trans.NormalizationTransformer(transform_X=True, dataset=train_ds)
    train_ds, valid_ds, test_ds = norm_X.transform(train_ds), norm_X.transform(valid_ds), norm_X.transform(test_ds)

    norm_y = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_ds)
    train_ds, valid_ds, test_ds = norm_y.transform(train_ds), norm_y.transform(valid_ds), norm_y.transform(test_ds)

    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = [
        '#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=',
        'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\',
        'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]',
        '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.',
        '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]'
    ]
    token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'] = 0
    token_to_id['<UNK>'] = 1

    def encode_smiles(smiles_list):
        encoded = []
        for s in smiles_list:
            tokens = tokenizer.tokenize(s)
            ids = [token_to_id.get(t, 1) for t in tokens[:MAX_LEN]]
            ids += [0] * (MAX_LEN - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    vocab_size = len(token_to_id)
    return train_ds, valid_ds, test_ds, encode_smiles, vocab_size

# ==============================================================================
# 3. Custom PyTorch Dataset & Collation (to handle Graphs)
# ==============================================================================
class FusionDataset(Dataset):
    def __init__(self, ds, encode_fn):
        self.smiles = ds.ids
        self.desc = torch.tensor(ds.X, dtype=torch.float32)
        self.y = torch.tensor(ds.y, dtype=torch.float32)
        self.smiles_ids = encode_fn(self.smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        node_feat, edge_idx = smiles_to_graph(smi)
        return self.smiles_ids[idx], self.desc[idx], node_feat, edge_idx, self.y[idx]

def custom_collate(batch):
    b_smiles, b_desc, b_nodes, b_edges, b_y = zip(*batch)
    
    smiles_t = torch.stack(b_smiles)
    desc_t = torch.stack(b_desc)
    y_t = torch.stack(b_y)
    
    batch_nodes = []
    batch_edges = []
    node_offset = 0
    batch_indices = []

    for i, (nodes, edges) in enumerate(zip(b_nodes, b_edges)):
        batch_nodes.append(torch.tensor(nodes, dtype=torch.float32))
        if edges.shape[1] > 0:
            edges_t = torch.tensor(edges, dtype=torch.long) + node_offset
            batch_edges.append(edges_t)
        batch_indices.append(torch.full((nodes.shape[0],), i, dtype=torch.long))
        node_offset += nodes.shape[0]

    nodes_t = torch.cat(batch_nodes, dim=0)
    edges_t = torch.cat(batch_edges, dim=1) if len(batch_edges) > 0 else torch.zeros((2, 0), dtype=torch.long)
    batch_idx_t = torch.cat(batch_indices, dim=0)

    return smiles_t, desc_t, nodes_t, edges_t, batch_idx_t, y_t

# ==============================================================================
# 4. Neural Network Architecture
# ==============================================================================
class StackedMamba(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + layer(x))
        return x.mean(dim=1)

class SimpleGraphBranch(nn.Module):
    def __init__(self, node_dim=10, hidden_dim=128):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.message_pass = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, x, edge_index, batch_idx, batch_size):
        x = self.node_proj(x)
        if edge_index.shape[1] > 0:
            row, col = edge_index
            messages = x[col]
            out = torch.zeros_like(x)
            out.index_add_(0, row, messages)
            x = x + self.message_pass(out)

        graph_embeds = torch.zeros(batch_size, x.shape[1], device=x.device)
        ones = torch.ones(batch_idx.size(0), 1, device=x.device)
        counts = torch.zeros(batch_size, 1, device=x.device)
        
        graph_embeds.index_add_(0, batch_idx, x)
        counts.index_add_(0, batch_idx, ones)
        graph_embeds = graph_embeds / counts.clamp(min=1)

        return self.out_proj(graph_embeds)

class WeightedFusionGate(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        # FIXED: Start all gates equally so RDKit doesn't dominate immediately
        with torch.no_grad():
            self.gate[-1].bias.data = torch.tensor([0.0, 0.0, 0.0])
        self.residual_proj = nn.Linear(dim * 3, dim)

    def forward(self, f_seq, f_phys, f_graph):
        combined = torch.cat([f_seq, f_phys, f_graph], dim=-1)
        gates = torch.softmax(self.gate(combined), dim=-1)
        f_weighted = (gates[:, 0:1] * f_seq + gates[:, 1:2] * f_phys + gates[:, 2:3] * f_graph)
        return f_weighted + 0.2 * self.residual_proj(combined), gates

class NeuroChemFusionV7(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = StackedMamba(d_model=128)
        
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        
        self.graph_branch = SimpleGraphBranch()

        self.fusion_gate = WeightedFusionGate(dim=128)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, smiles_ids, descriptors, nodes, edges, batch_idx, return_gates=False):
        batch_size = smiles_ids.shape[0]
        
        f_seq   = self.mamba_branch(self.embedding(smiles_ids))
        f_phys  = self.phys_branch(descriptors)
        f_graph = self.graph_branch(nodes, edges, batch_idx, batch_size)

        f_fusion, gates = self.fusion_gate(f_seq, f_phys, f_graph)
        out = self.regressor(f_fusion)

        if return_gates:
            return out, gates
        return out

# ==============================================================================
# 5. Training Loop
# ==============================================================================
train_ds, valid_ds, test_ds, encode_fn, vocab_size = prepare_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(FusionDataset(train_ds, encode_fn), batch_size=64, shuffle=True, collate_fn=custom_collate)
valid_loader = DataLoader(FusionDataset(valid_ds, encode_fn), batch_size=64, shuffle=False, collate_fn=custom_collate)
test_loader  = DataLoader(FusionDataset(test_ds,  encode_fn), batch_size=64, shuffle=False, collate_fn=custom_collate)

model = NeuroChemFusionV7(vocab_size=vocab_size, desc_dim=train_ds.X.shape[1]).to(device)

# Differential Optimizer: Give GNN and Mamba a higher learning rate so they learn fast enough
optimizer = torch.optim.AdamW([
    {'params': model.mamba_branch.parameters(), 'lr': 1e-3},
    {'params': model.embedding.parameters(), 'lr': 1e-3},
    {'params': model.graph_branch.parameters(), 'lr': 1e-3},
    {'params': model.phys_branch.parameters(), 'lr': 5e-4}, # RDKit learns easily, needs less LR
    {'params': model.fusion_gate.parameters(), 'lr': 5e-4},
    {'params': model.regressor.parameters(), 'lr': 5e-4},
], weight_decay=1e-4)

# Added scheduler back to stabilize training
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[1e-3, 1e-3, 1e-3, 5e-4, 5e-4, 5e-4], 
    epochs=100, steps_per_epoch=len(train_loader), pct_start=0.1
)

criterion = nn.HuberLoss(delta=1.0)

print("\n>>> Starting End-to-End Fusion Training...", flush=True)

best_r2 = -float('inf')
for epoch in range(1, 101):
    model.train()
    epoch_loss = 0.0
    for b_smiles, b_desc, b_nodes, b_edges, b_batch_idx, b_y in train_loader:
        b_smiles, b_desc, b_y = b_smiles.to(device), b_desc.to(device), b_y.to(device)
        b_nodes, b_edges, b_batch_idx = b_nodes.to(device), b_edges.to(device), b_batch_idx.to(device)

        optimizer.zero_grad()
        preds = model(b_smiles, b_desc, b_nodes, b_edges, b_batch_idx)
        loss = criterion(preds, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    if epoch % 5 == 0:
        model.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for b_smiles, b_desc, b_nodes, b_edges, b_batch_idx, b_y in valid_loader:
                b_smiles, b_desc = b_smiles.to(device), b_desc.to(device)
                b_nodes, b_edges, b_batch_idx = b_nodes.to(device), b_edges.to(device), b_batch_idx.to(device)
                preds = model(b_smiles, b_desc, b_nodes, b_edges, b_batch_idx)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_y.extend(b_y.numpy().flatten())
                
        v_r2 = pearsonr(all_preds, all_y)[0] ** 2
        print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(train_loader):.4f} | Val R²: {v_r2:.4f}")
        
        if v_r2 > best_r2:
            best_r2 = v_r2
            torch.save(model.state_dict(), 'neurochem_v7_best.pt')

print("\n>>> Training Complete! All 3 features working end-to-end.")

# ==============================================================================
# 6. Final Evaluation on Test Set
# ==============================================================================
print("\n>>> Loading Best Model for Testing...", flush=True)
model.load_state_dict(torch.load('neurochem_v7_best.pt'))
model.eval()

test_preds = []
test_trues = []
all_gates = []

with torch.no_grad():
    for b_smiles, b_desc, b_nodes, b_edges, b_batch_idx, b_y in test_loader:
        b_smiles = b_smiles.to(device)
        b_desc = b_desc.to(device)
        b_nodes = b_nodes.to(device)
        b_edges = b_edges.to(device)
        b_batch_idx = b_batch_idx.to(device)
        
        out, gates = model(b_smiles, b_desc, b_nodes, b_edges, b_batch_idx, return_gates=True)
        
        test_preds.extend(out.cpu().numpy().flatten())
        test_trues.extend(b_y.numpy().flatten())
        all_gates.append(gates.cpu().numpy())

test_preds = np.array(test_preds)
test_trues = np.array(test_trues)
all_gates = np.concatenate(all_gates, axis=0)

final_r2 = pearsonr(test_preds, test_trues)[0] ** 2
final_mse = np.mean((test_preds - test_trues) ** 2)
avg_gates = all_gates.mean(axis=0)

print("\n" + "="*50)
print("NEUROCHEM FUSION V7 (END-TO-END) - FINAL RESULTS")
print(f"Test Pearson R²: {final_r2:.4f}")
print(f"Test MSE:        {final_mse:.4f}")
print(f"Branch weights  → seq(Mamba): {avg_gates[0]:.3f} | phys(RDKit): {avg_gates[1]:.3f} | graph(GNN): {avg_gates[2]:.3f}")
print("="*50, flush=True)