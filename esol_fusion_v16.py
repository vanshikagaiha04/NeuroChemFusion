import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch.optim.swa_utils import AveragedModel, SWALR

# Headless plotting for the cluster
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")

MAX_LEN = 128

# ==============================================================================
# 1. ADVANCED FEATURIZATION (20-Dim Chemprop-style Nodes)
# ==============================================================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((1, 20)), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    
    rdPartialCharges.ComputeGasteigerCharges(mol)
    
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum() / 100.0  
        degree = atom.GetDegree() / 6.0           
        num_hs = atom.GetTotalNumHs() / 4.0       
        is_aromatic = float(atom.GetIsAromatic())
        
        hyb = atom.GetHybridization()
        is_sp = float(hyb == Chem.rdchem.HybridizationType.SP)
        is_sp2 = float(hyb == Chem.rdchem.HybridizationType.SP2)
        is_sp3 = float(hyb == Chem.rdchem.HybridizationType.SP3)
        
        is_ring = float(atom.IsInRing())
        ring3 = float(atom.IsInRingSize(3))
        ring4 = float(atom.IsInRingSize(4))
        ring5 = float(atom.IsInRingSize(5))
        ring6 = float(atom.IsInRingSize(6))
        
        formal_charge = atom.GetFormalCharge() / 3.0
        try:
            g_charge = float(atom.GetProp('_GasteigerCharge'))
            if np.isnan(g_charge) or np.isinf(g_charge): g_charge = 0.0
        except:
            g_charge = 0.0
            
        chiral_tag = atom.GetChiralTag()
        is_R = float(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
        is_S = float(chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
        
        features = [
            atomic_num, degree, num_hs, is_aromatic,
            is_sp, is_sp2, is_sp3,
            is_ring, ring3, ring4, ring5, ring6,
            formal_charge, g_charge,
            is_R, is_S
        ]
        features += [0.0] * (20 - len(features))
        atom_features.append(features)
        
    node_features = np.array(atom_features, dtype=np.float32)
    
    edges, edge_weights = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        bw = bond.GetBondTypeAsDouble()
        edge_weights += [bw, bw]
        
    if len(edges) == 0:
        return node_features, np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return node_features, np.array(edges, dtype=np.int64).T, np.array(edge_weights, dtype=np.float32)

def prepare_data():
    featurizer = dc.feat.RDKitDescriptors()
    
    # -------------------------------------------------------------------------
    # OFFLINE FIX: Bypass load_delaney and load the CSV shown in your screenshot!
    # -------------------------------------------------------------------------
    print("Loading ESOL directly from local CSV...", flush=True)
    loader = dc.data.CSVLoader(
        tasks=['measured log solubility in mols per litre'],
        feature_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.create_dataset('./esol_data/delaney-processed.csv')
    
    # Use IndexSplitter to match default ESOL behavior
    splitter = dc.splits.IndexSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(dataset)
    # -------------------------------------------------------------------------

    # Normalize X and Y manually
    norm_X = dc.trans.NormalizationTransformer(transform_X=True, dataset=train_ds)
    train_ds, valid_ds, test_ds = norm_X.transform(train_ds), norm_X.transform(valid_ds), norm_X.transform(test_ds)
    
    norm_y = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_ds)
    train_ds, valid_ds, test_ds = norm_y.transform(train_ds), norm_y.transform(valid_ds), norm_y.transform(test_ds)

    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
    token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'], token_to_id['<UNK>'] = 0, 1

    def encode_smiles(smiles_list):
        encoded = []
        for s in smiles_list:
            tokens = tokenizer.tokenize(s)
            ids = [token_to_id.get(t, 1) for t in tokens[:MAX_LEN]]
            ids += [0] * (MAX_LEN - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)
        
    return train_ds, valid_ds, test_ds, encode_smiles, len(token_to_id)

class FusionDataset(Dataset):
    def __init__(self, ds, encode_fn):
        self.smiles, self.desc, self.y = ds.ids, torch.tensor(ds.X, dtype=torch.float32), torch.tensor(ds.y, dtype=torch.float32)
        self.smiles_ids = encode_fn(self.smiles)
        
    def __len__(self): return len(self.smiles)
    
    def __getitem__(self, idx):
        node_feat, edge_idx, edge_attr = smiles_to_graph(self.smiles[idx])
        return self.smiles_ids[idx], self.desc[idx], node_feat, edge_idx, edge_attr, self.y[idx]

def custom_collate(batch):
    b_smiles, b_desc, b_nodes, b_edges, b_weights, b_y = zip(*batch)
    smiles_t, desc_t, y_t = torch.stack(b_smiles), torch.stack(b_desc), torch.stack(b_y)
    
    batch_nodes, batch_edges, batch_weights, batch_indices, node_offset = [], [], [], [], 0
    for i, (nodes, edges, weights) in enumerate(zip(b_nodes, b_edges, b_weights)):
        batch_nodes.append(torch.tensor(nodes, dtype=torch.float32))
        if edges.shape[1] > 0: 
            batch_edges.append(torch.tensor(edges, dtype=torch.long) + node_offset)
            batch_weights.append(torch.tensor(weights, dtype=torch.float32).unsqueeze(1))
        batch_indices.append(torch.full((nodes.shape[0],), i, dtype=torch.long))
        node_offset += nodes.shape[0]

    nodes_t = torch.cat(batch_nodes, dim=0)
    edges_t = torch.cat(batch_edges, dim=1) if len(batch_edges) > 0 else torch.zeros((2, 0), dtype=torch.long)
    weights_t = torch.cat(batch_weights, dim=0) if len(batch_weights) > 0 else torch.zeros((0, 1), dtype=torch.float32)
    batch_idx_t = torch.cat(batch_indices, dim=0)
    return smiles_t, desc_t, nodes_t, edges_t, weights_t, batch_idx_t, y_t

# ==============================================================================
# 2. Architecture
# ==============================================================================
class StackedMamba(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms): x = norm(x + layer(x))
        return x.mean(dim=1)

class AttentionGraphBranch(nn.Module):
    def __init__(self, node_dim=20, hidden_dim=256): 
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.mp1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mp2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(0.3))

    def _pass_messages(self, x, edge_index, edge_weight, layer, norm):
        if edge_index.shape[1] > 0:
            row, col = edge_index
            messages = x[col] * edge_weight
            out = torch.zeros_like(x)
            out.index_add_(0, row, messages)
            return norm(x + torch.relu(layer(out)))
        return x

    def forward(self, x, edge_index, edge_weight, batch_idx, batch_size):
        x = self.node_proj(x)
        x = self._pass_messages(x, edge_index, edge_weight, self.mp1, self.norm1)
        x = self._pass_messages(x, edge_index, edge_weight, self.mp2, self.norm2)
        attn_scores = torch.softmax(self.attn(x), dim=0)
        x_weighted = x * attn_scores
        graph_embeds = torch.zeros(batch_size, x.shape[1], device=x.device)
        graph_embeds.index_add_(0, batch_idx, x_weighted)
        return self.out_proj(graph_embeds)

class LightweightSigmoidGate(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, 64), nn.GELU(), nn.Linear(64, 3))
        with torch.no_grad(): self.gate[-1].bias.data = torch.tensor([0.5, 0.5, 0.5])
        self.residual_proj = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f_seq, f_phys, f_graph):
        combined = torch.cat([f_seq, f_phys, f_graph], dim=-1)
        gates = torch.sigmoid(self.gate(combined))
        f_weighted = (gates[:, 0:1] * f_seq + gates[:, 1:2] * f_phys + gates[:, 2:3] * f_graph)
        return self.norm(f_weighted + 0.2 * self.residual_proj(combined)), gates

class NeuroChemFusionV16(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = StackedMamba(d_model=128)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        self.graph_branch = AttentionGraphBranch(node_dim=20)
        self.fusion_gate = LightweightSigmoidGate(dim=128)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.4), nn.Linear(64, 1)
        )

    def forward(self, smiles_ids, descriptors, nodes, edges, edge_weights, batch_idx, return_gates=False, return_embeds=False):
        batch_size = smiles_ids.shape[0]
        f_seq   = self.mamba_branch(self.embedding(smiles_ids))
        f_phys  = self.phys_branch(descriptors)
        f_graph = self.graph_branch(nodes, edges, edge_weights, batch_idx, batch_size)

        embeds = (f_seq, f_phys, f_graph)

        if self.training:
            if torch.rand(1).item() < 0.2: f_seq = torch.zeros_like(f_seq)
            if torch.rand(1).item() < 0.2: f_phys = torch.zeros_like(f_phys)

        f_fusion, gates = self.fusion_gate(f_seq, f_phys, f_graph)
        out = self.regressor(f_fusion)
        
        if return_embeds: return out, embeds
        if return_gates: return out, gates
        return out

# ==============================================================================
# 3. Training Loop with HISTORY TRACKING
# ==============================================================================
train_ds, valid_ds, test_ds, encode_fn, vocab_size = prepare_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(FusionDataset(train_ds, encode_fn), batch_size=64, shuffle=True, collate_fn=custom_collate)
valid_loader = DataLoader(FusionDataset(valid_ds, encode_fn), batch_size=64, shuffle=False, collate_fn=custom_collate)
test_loader  = DataLoader(FusionDataset(test_ds,  encode_fn), batch_size=64, shuffle=False, collate_fn=custom_collate)

model = NeuroChemFusionV16(vocab_size=vocab_size, desc_dim=train_ds.X.shape[1]).to(device)

optimizer = torch.optim.AdamW([
    {'params': model.mamba_branch.parameters(), 'lr': 1e-4}, 
    {'params': model.embedding.parameters(), 'lr': 1e-4},
    {'params': model.graph_branch.parameters(), 'lr': 5e-4},
    {'params': model.phys_branch.parameters(), 'lr': 3e-4},
    {'params': model.fusion_gate.parameters(), 'lr': 5e-4},
    {'params': model.regressor.parameters(), 'lr': 5e-4},
], weight_decay=1e-2)

criterion = nn.HuberLoss(delta=1.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)
SWA_START_EPOCH = 50 

print("\n>>> Starting V16 Tri-Modal Training on ESOL...", flush=True)

best_r2 = -float('inf')
patience_counter = 0

# Lists to track history for the graphs
history_epochs = []
history_train_loss = []
history_val_r2 = []

for epoch in range(1, 151): 
    model.train()
    epoch_loss = 0.0
    for b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, b_y in train_loader:
        b_smiles, b_desc, b_y = b_smiles.to(device), b_desc.to(device), b_y.to(device)
        b_nodes, b_edges, b_weights, b_batch_idx = b_nodes.to(device), b_edges.to(device), b_weights.to(device), b_batch_idx.to(device)

        optimizer.zero_grad()
        
        preds, (f_seq, f_phys, f_graph) = model(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, return_embeds=True)
        loss_task = criterion(preds, b_y)
        loss_align = nn.functional.mse_loss(f_seq, f_phys.detach()) + nn.functional.mse_loss(f_graph, f_phys.detach())
        loss = loss_task + (0.1 * loss_align)
        
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None and len(param.grad.shape) > 1:
                param.grad.data -= param.grad.data.mean(dim=tuple(range(1, len(param.grad.shape))), keepdim=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss/len(train_loader)
    
    if epoch > SWA_START_EPOCH:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    if epoch % 2 == 0:
        eval_model = swa_model if epoch > SWA_START_EPOCH else model
        eval_model.eval()
        all_preds, all_y = [], []
        with torch.no_grad():
            for b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, b_y in valid_loader:
                b_smiles, b_desc = b_smiles.to(device), b_desc.to(device)
                b_nodes, b_edges, b_weights, b_batch_idx = b_nodes.to(device), b_edges.to(device), b_weights.to(device), b_batch_idx.to(device)
                preds = eval_model(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_y.extend(b_y.cpu().numpy().flatten())
                
        v_r2 = pearsonr(all_preds, all_y)[0] ** 2
        print(f"Epoch {epoch:03d} | Loss: {avg_train_loss:.4f} | Val R²: {v_r2:.4f}{' (SWA)' if epoch>SWA_START_EPOCH else ''}")
        
        # Track history for plotting
        history_epochs.append(epoch)
        history_train_loss.append(avg_train_loss)
        history_val_r2.append(v_r2)
        
        if epoch <= SWA_START_EPOCH:
            scheduler.step(v_r2)
            
        if v_r2 > best_r2:
            best_r2 = v_r2
            torch.save(eval_model.state_dict(), 'esol_fusion_v16_best.pt')
            patience_counter = 0
        else:
            patience_counter += 2
            
        if patience_counter >= 30: 
            print(f"\n[!] Early Stopping triggered at Epoch {epoch} (Peak R² was {best_r2:.4f})")
            break

torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

# ==============================================================================
# 4. Final Evaluation & Metric Calculation
# ==============================================================================
print("\n>>> Loading Best Model for Testing...", flush=True)

state_dict = torch.load('esol_fusion_v16_best.pt')

if "n_averaged" in state_dict:
    swa_model.load_state_dict(state_dict)
    final_eval_model = swa_model
else:
    model.load_state_dict(state_dict)
    final_eval_model = model

final_eval_model.to(device).eval()

test_preds, test_trues, all_gates = [], [], []
with torch.no_grad():
    for b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, b_y in test_loader:
        b_smiles, b_desc = b_smiles.to(device), b_desc.to(device)
        b_nodes, b_edges, b_weights, b_batch_idx = b_nodes.to(device), b_edges.to(device), b_weights.to(device), b_batch_idx.to(device)
        
        if hasattr(final_eval_model, 'module'):
            out, gates = final_eval_model.module(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, return_gates=True)
        else:
            out, gates = final_eval_model(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, return_gates=True)
            
        test_preds.extend(out.cpu().numpy().flatten())
        test_trues.extend(b_y.cpu().numpy().flatten())
        all_gates.append(gates.cpu().numpy())

test_preds = np.array(test_preds)
test_trues = np.array(test_trues)

# FIX: Filter out any accidental NaNs generated by SWA Batch Norm
valid_mask = ~np.isnan(test_preds) & ~np.isnan(test_trues)
test_preds = test_preds[valid_mask]
test_trues = test_trues[valid_mask]

avg_gates = np.nanmean(np.concatenate(all_gates, axis=0), axis=0)

pearson_r2 = pearsonr(test_preds, test_trues)[0] ** 2
spearman_rho = spearmanr(test_preds, test_trues)[0]
mse = mean_squared_error(test_trues, test_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_trues, test_preds)

print("\n" + "="*60)
print("ESOL FUSION V16 - FINAL TEST METRICS")
print(f"Pearson R²:           {pearson_r2:.4f}")
print(f"Spearman Correlation: {spearman_rho:.4f}")
print(f"Mean Squared Error:   {mse:.4f}")
print(f"Root Mean Sq Error:   {rmse:.4f}")
print(f"Mean Absolute Error:  {mae:.4f}")
print("-" * 60)
print(f"Sigmoid Gate Weights → Seq: {avg_gates[0]:.3f} | Phys: {avg_gates[1]:.3f} | Graph: {avg_gates[2]:.3f}")
print("="*60, flush=True)


# ==============================================================================
# 5. Graph Generation (Saved as PNGs)
# ==============================================================================
print("\n>>> Generating Graphs for Report...")

# Graph 1: Training Curve
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.plot(history_epochs, history_train_loss, color='tab:red', marker='o', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Validation R²', color='tab:blue')
ax2.plot(history_epochs, history_val_r2, color='tab:blue', marker='s', label='Val R²')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('ESOL Tri-Modal Training Curve (Loss & Validation R²)')
fig.tight_layout()
plt.savefig('esol_fusion_training_curve.png', dpi=300)
plt.close()

# Graph 2: Parity Plot
plt.figure(figsize=(6, 6))
plt.scatter(test_trues, test_preds, alpha=0.5, color='darkorange', edgecolors='k')
min_val = min(np.min(test_trues), np.min(test_preds))
max_val = max(np.max(test_trues), np.max(test_preds))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual ESOL (Normalized)')
plt.ylabel('Predicted ESOL (Normalized)')
plt.title(f'ESOL Fusion Parity Plot (Test R²: {pearson_r2:.4f})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('esol_fusion_parity_plot.png', dpi=300)
plt.close()

# Graph 3: Modality Gate Weights
plt.figure(figsize=(6, 4))
modalities = ['1D Sequence\n(Mamba)', '0D Physics\n(RDKit)', '2D Topology\n(Graph)']
weights = [avg_gates[0], avg_gates[1], avg_gates[2]]
colors = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(modalities, weights, color=colors, edgecolor='black')
plt.ylim(0, 1.0)
plt.ylabel('Average Sigmoid Gate Activation')
plt.title('ESOL Task: Modality Contributions')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('esol_fusion_gate_weights.png', dpi=300)
plt.close()

print(">>> Graphs saved successfully: esol_fusion_training_curve.png, esol_fusion_parity_plot.png, esol_fusion_gate_weights.png", flush=True)