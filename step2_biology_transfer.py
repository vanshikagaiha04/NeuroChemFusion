import os, torch, warnings
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

# ==========================================
# 1. IDENTICAL DATA PREP
# ==========================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((1, 20)), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 100.0, atom.GetDegree() / 6.0, atom.GetTotalNumHs() / 4.0, float(atom.GetIsAromatic()),
            float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP), float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2), float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
            float(atom.IsInRing()), float(atom.IsInRingSize(3)), float(atom.IsInRingSize(4)), float(atom.IsInRingSize(5)), float(atom.IsInRingSize(6)),
            atom.GetFormalCharge() / 3.0, 0.0, float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW), float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)
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
    if len(edges) == 0: return node_features, np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return node_features, np.array(edges, dtype=np.int64).T, np.array(edge_weights, dtype=np.float32)

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

class ClassificationDataset(Dataset):
    def __init__(self, ds):
        self.smiles = ds.ids
        # FIX 1: Robust NaN and Inf handling
        clean_X = np.nan_to_num(ds.X.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        self.desc = torch.tensor(clean_X, dtype=torch.float32)
        
        self.y = torch.tensor(np.nan_to_num(ds.y.astype(np.float32)), dtype=torch.float32)
        self.w = torch.tensor(np.nan_to_num(ds.w.astype(np.float32)), dtype=torch.float32) if hasattr(ds, 'w') else torch.ones_like(self.y)
        self.smiles_ids = encode_smiles(self.smiles)
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx):
        node_feat, edge_idx, edge_attr = smiles_to_graph(self.smiles[idx])
        return self.smiles_ids[idx], self.desc[idx], node_feat, edge_idx, edge_attr, self.y[idx], self.w[idx]

def custom_collate_cls(batch):
    b_smiles, b_desc, b_nodes, b_edges, b_weights, b_y, b_w = zip(*batch)
    smiles_t, desc_t, y_t, w_t = torch.stack(b_smiles), torch.stack(b_desc), torch.stack(b_y), torch.stack(b_w)
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
    return smiles_t, desc_t, nodes_t, edges_t, weights_t, batch_idx_t, y_t, w_t

# ==========================================
# 2. IDENTICAL BACKBONE + TRANSFER HEAD
# ==========================================
class StackedMamba(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.layers = nn.ModuleList([Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
    def forward(self, x):
        for l, n in zip(self.layers, self.norms): x = n(x + l(x))
        return x.mean(dim=1)

class AttentionGraphBranch(nn.Module):
    def __init__(self): 
        super().__init__()
        self.node_proj = nn.Linear(20, 256)
        self.mp1 = nn.Linear(256, 256)
        self.norm1 = nn.LayerNorm(256)
        self.attn = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out_proj = nn.Sequential(nn.Linear(256, 128), nn.GELU())
    def forward(self, x, edge_index, edge_weight, batch_idx, batch_size):
        x = self.node_proj(x)
        if edge_index.shape[1] > 0:
            row, col = edge_index
            messages = x[col] * edge_weight
            out = torch.zeros_like(x).index_add_(0, row, messages)
            x = self.norm1(x + torch.relu(self.mp1(out)))
        attn_scores = torch.softmax(self.attn(x), dim=0)
        graph_embeds = torch.zeros(batch_size, x.shape[1], device=x.device).index_add_(0, batch_idx, x * attn_scores)
        return self.out_proj(graph_embeds)

class NeuroChemBackbone(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = StackedMamba()
        self.phys_branch = nn.Sequential(nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128))
        self.graph_branch = AttentionGraphBranch()
        self.gate = nn.Sequential(nn.Linear(128 * 3, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, s, d, n, e, ew, b_idx):
        f_seq = self.mamba_branch(self.embedding(s))
        f_phys = self.phys_branch(d)
        f_graph = self.graph_branch(n, e, ew, b_idx, s.shape[0])
        gates = torch.sigmoid(self.gate(torch.cat([f_seq, f_phys, f_graph], dim=-1)))
        return (gates[:, 0:1]*f_seq + gates[:, 1:2]*f_phys + gates[:, 2:3]*f_graph)

class TransferBiologyModel(nn.Module):
    def __init__(self, backbone, n_tasks):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.5), nn.Linear(64, n_tasks))
    def forward(self, s, d, n, e, ew, b_idx):
        features = self.backbone(s, d, n, e, ew, b_idx)
        return self.head(features)

# ==========================================
# 3. TRANSFER TRAINING
# ==========================================
def train_biology_task(task_name, n_tasks, data_dir, epochs=40):
    print(f"\n{'='*50}\n>>> STAGE: Biology Transfer on {task_name.upper()}\n{'='*50}", flush=True)
    f_desc = dc.feat.RDKitDescriptors()
    
    if task_name == 'bbbp':
        _, ds, _ = dc.molnet.load_bbbp(featurizer=f_desc, data_dir=data_dir, reload=False)
    else:
        _, ds, _ = dc.molnet.load_tox21(featurizer=f_desc, data_dir=data_dir, reload=False)
        
    train_ds, valid_ds, test_ds = dc.splits.ScaffoldSplitter().train_valid_test_split(ds[0])
    
    train_loader = DataLoader(ClassificationDataset(train_ds), batch_size=64, shuffle=True, collate_fn=custom_collate_cls)
    test_loader = DataLoader(ClassificationDataset(test_ds), batch_size=64, shuffle=False, collate_fn=custom_collate_cls)

    backbone = NeuroChemBackbone(len(token_to_id)+2, desc_dim=217)
    print(">>> Loading pre-trained Chemistry Brain...", flush=True)
    backbone.load_state_dict(torch.load('master_chemistry_backbone.pt'))
    
    model = TransferBiologyModel(backbone, n_tasks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print(f">>> Training {task_name.upper()} Head...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for s, d, n, e, ew, b_idx, y, w in train_loader:
            s, d, n, e, ew, b_idx, y, w = s.to(device), d.to(device), n.to(device), e.to(device), ew.to(device), b_idx.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            preds = model(s, d, n, e, ew, b_idx)
            loss = (criterion(preds, y) * w).mean()
            loss.backward()
            
            # FIX 2: Gradient Clipping to prevent NaN explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    # Evaluation
    model.eval()
    all_preds, all_trues, all_weights = [], [], []
    with torch.no_grad():
        for s, d, n, e, ew, b_idx, y, w in test_loader:
            s, d, n, e, ew, b_idx = s.to(device), d.to(device), n.to(device), e.to(device), ew.to(device), b_idx.to(device)
            p = torch.sigmoid(model(s, d, n, e, ew, b_idx))
            
            # Safeguard against NaNs in predictions just in case
            p = torch.nan_to_num(p, nan=0.5) 
            
            all_preds.extend(p.cpu().numpy())
            all_trues.extend(y.numpy())
            all_weights.extend(w.numpy())

    all_preds, all_trues, all_weights = np.array(all_preds), np.array(all_trues), np.array(all_weights)
    
    # Calculate AUC
    if n_tasks == 1: 
        auc = roc_auc_score(all_trues, all_preds)
        print(f"\n[FINAL TEST] {task_name.upper()} ROC-AUC = {auc:.4f}", flush=True)
    else: 
        # FIX 3: Proper Task-by-Task AUC for Tox21
        task_aucs = []
        for i in range(n_tasks):
            valid = all_weights[:, i] > 0
            if valid.sum() > 0 and len(np.unique(all_trues[valid, i])) > 1:
                task_auc = roc_auc_score(all_trues[valid, i], all_preds[valid, i])
                task_aucs.append(task_auc)
        
        auc = np.mean(task_aucs)
        print(f"\n[FINAL TEST] {task_name.upper()} Average ROC-AUC = {auc:.4f}", flush=True)

    torch.save(model.state_dict(), f'final_{task_name}_model.pt')
    print(f">>> {task_name.upper()} Model Saved!", flush=True)

if __name__ == '__main__':
    # 1. Train BBBP
    train_biology_task('bbbp', n_tasks=1, data_dir='./bbbp_data_desc', epochs=40)
    
    # 2. Train Tox21
    train_biology_task('tox21', n_tasks=12, data_dir='./tox21_data_desc', epochs=50)
    
    print("\n>>> ALL TASKS FINISHED! Your Tri-Modal Unified Pipeline is Complete! 🏆", flush=True)