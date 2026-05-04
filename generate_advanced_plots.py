import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, matthews_corrcoef
)
import warnings
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from torch.optim.swa_utils import AveragedModel

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")

MAX_LEN = 128

# ==============================================================================
# 1. FEATURIZATION & OFFLINE DATA LOADING
# ==============================================================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((1, 20)), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 100.0, atom.GetDegree() / 6.0, atom.GetTotalNumHs() / 4.0, float(atom.GetIsAromatic()),
            float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
            float(atom.IsInRing()), float(atom.IsInRingSize(3)), float(atom.IsInRingSize(4)),
            float(atom.IsInRingSize(5)), float(atom.IsInRingSize(6)), atom.GetFormalCharge() / 3.0
        ]
        try: g_charge = float(atom.GetProp('_GasteigerCharge'))
        except: g_charge = 0.0
        features.extend([g_charge if not np.isnan(g_charge) else 0.0,
                         float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
                         float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)])
        features += [0.0] * (20 - len(features))
        atom_features.append(features)
    edges, edge_weights = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        bw = bond.GetBondTypeAsDouble()
        edge_weights += [bw, bw]
    if not edges: return np.array(atom_features, dtype=np.float32), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return np.array(atom_features, dtype=np.float32), np.array(edges, dtype=np.int64).T, np.array(edge_weights, dtype=np.float32)

def prepare_data():
    featurizer = dc.feat.RDKitDescriptors()
    
    print(">>> Loading ESOL offline for Evaluation...", flush=True)
    loader = dc.data.CSVLoader(
        tasks=['measured log solubility in mols per litre'],
        feature_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.create_dataset('./esol_data/delaney-processed.csv')
    
    splitter = dc.splits.IndexSplitter()
    train_ds, valid_ds, test_ds = splitter.train_valid_test_split(dataset)

    norm_X = dc.trans.NormalizationTransformer(transform_X=True, dataset=train_ds)
    test_ds = norm_X.transform(test_ds)
    norm_y = dc.trans.NormalizationTransformer(transform_y=True, dataset=train_ds)
    test_ds = norm_y.transform(test_ds)

    tokenizer = dc.feat.BasicSmilesTokenizer()
    unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
    token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
    token_to_id['<PAD>'], token_to_id['<UNK>'] = 0, 1
    def encode_smiles(smiles_list):
        encoded = []
        for s in smiles_list:
            ids = [token_to_id.get(t, 1) for t in tokenizer.tokenize(s)[:MAX_LEN]]
            encoded.append(ids + [0] * (MAX_LEN - len(ids)))
        return torch.tensor(encoded, dtype=torch.long)
    return test_ds, encode_smiles, len(token_to_id)

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
    return smiles_t, desc_t, nodes_t, edges_t, weights_t, torch.cat(batch_indices, dim=0), y_t

# ==============================================================================
# 2. ARCHITECTURE
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
        self.node_proj, self.mp1, self.norm1 = nn.Linear(node_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        self.mp2, self.norm2 = nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        self.attn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.out_proj = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(0.3))
    def _pass_messages(self, x, edge_index, edge_weight, layer, norm):
        if edge_index.shape[1] > 0:
            out = torch.zeros_like(x).index_add_(0, edge_index[0], x[edge_index[1]] * edge_weight)
            return norm(x + torch.relu(layer(out)))
        return x
    def forward(self, x, edge_index, edge_weight, batch_idx, batch_size):
        x = self._pass_messages(self.node_proj(x), edge_index, edge_weight, self.mp1, self.norm1)
        x = self._pass_messages(x, edge_index, edge_weight, self.mp2, self.norm2)
        graph_embeds = torch.zeros(batch_size, x.shape[1], device=x.device).index_add_(0, batch_idx, x * torch.softmax(self.attn(x), dim=0))
        return self.out_proj(graph_embeds)

class LightweightSigmoidGate(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate, self.residual_proj, self.norm = nn.Sequential(nn.Linear(dim*3, 64), nn.GELU(), nn.Linear(64, 3)), nn.Linear(dim*3, dim), nn.LayerNorm(dim)
    def forward(self, f_seq, f_phys, f_graph):
        combined = torch.cat([f_seq, f_phys, f_graph], dim=-1)
        gates = torch.sigmoid(self.gate(combined))
        return self.norm((gates[:, 0:1]*f_seq + gates[:, 1:2]*f_phys + gates[:, 2:3]*f_graph) + 0.2*self.residual_proj(combined)), gates

class NeuroChemFusionV16(nn.Module):
    def __init__(self, vocab_size, desc_dim):
        super().__init__()
        self.embedding, self.mamba_branch, self.graph_branch = nn.Embedding(vocab_size, 128, padding_idx=0), StackedMamba(), AttentionGraphBranch()
        self.phys_branch = nn.Sequential(nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.LayerNorm(128))
        self.fusion_gate = LightweightSigmoidGate(128)
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.4), nn.Linear(64, 1))
    def forward(self, smiles_ids, descriptors, nodes, edges, edge_weights, batch_idx):
        f_seq, f_phys, f_graph = self.mamba_branch(self.embedding(smiles_ids)), self.phys_branch(descriptors), self.graph_branch(nodes, edges, edge_weights, batch_idx, smiles_ids.shape[0])
        f_fusion, _ = self.fusion_gate(f_seq, f_phys, f_graph)
        return self.regressor(f_fusion)

# ==============================================================================
# 3. EVALUATION & CLASSIFICATION THRESHOLDING
# ==============================================================================
print(">>> Initializing Model...", flush=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds, encode_fn, vocab_size = prepare_data()
test_loader = DataLoader(FusionDataset(test_ds, encode_fn), batch_size=64, shuffle=False, collate_fn=custom_collate)

model = NeuroChemFusionV16(vocab_size=vocab_size, desc_dim=test_ds.X.shape[1])

print(">>> Loading esol_fusion_v16_best.pt...", flush=True)
state_dict = torch.load('esol_fusion_v16_best.pt', map_location=device)

if "n_averaged" in state_dict:
    swa_model = AveragedModel(model)
    swa_model.load_state_dict(state_dict)
    model = swa_model
else:
    model.load_state_dict(state_dict)

model.to(device).eval()

print(">>> Running Inference...", flush=True)
test_preds, test_trues = [], []
with torch.no_grad():
    for b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx, b_y in test_loader:
        b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx = b_smiles.to(device), b_desc.to(device), b_nodes.to(device), b_edges.to(device), b_weights.to(device), b_batch_idx.to(device)
        out = model.module(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx) if hasattr(model, 'module') else model(b_smiles, b_desc, b_nodes, b_edges, b_weights, b_batch_idx)
        test_preds.extend(out.cpu().numpy().flatten())
        test_trues.extend(b_y.numpy().flatten())

# Remove NaNs
test_preds, test_trues = np.array(test_preds), np.array(test_trues)
valid_mask = ~np.isnan(test_preds) & ~np.isnan(test_trues)
test_preds, test_trues = test_preds[valid_mask], test_trues[valid_mask]

# --- BINARIZE FOR CLASSIFICATION ---
threshold = np.median(test_trues)
y_true_bin = (test_trues > threshold).astype(int)
y_pred_bin = (test_preds > threshold).astype(int)

# --- CALCULATE METRICS ---
cm = confusion_matrix(y_true_bin, y_pred_bin)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_true_bin, y_pred_bin)
f1 = f1_score(y_true_bin, y_pred_bin)
mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn) # Recall

fpr, tpr, _ = roc_curve(y_true_bin, test_preds)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_true_bin, test_preds)
pr_auc = auc(recall, precision)

print("\n" + "="*60)
print("ADVANCED CLASSIFICATION METRICS (BINARIZED ESOL)")
print("="*60)
print(f"Median Split Threshold:  {threshold:.4f} (Normalized)")
print("-" * 60)
print(f"ROC-AUC:                 {roc_auc:.4f}")
print(f"PR-AUC:                  {pr_auc:.4f}")
print(f"Accuracy:                {acc:.4f}")
print(f"F1-Score:                {f1:.4f}")
print(f"Matthews Corr Coef (MCC):{mcc:.4f}")
print(f"Sensitivity (Recall):    {sensitivity:.4f}")
print(f"Specificity (TNR):       {specificity:.4f}")
print("="*60 + "\n")

# ==============================================================================
# 4. PLOTTING
# ==============================================================================
print(">>> Generating Advanced Classification Plots...", flush=True)

# PLOT 1: Score Distribution
plt.figure(figsize=(7, 5))
plt.hist(test_trues, bins=30, alpha=0.6, color='blue', label='Actual ESOL', density=True)
plt.hist(test_preds, bins=30, alpha=0.6, color='orange', label='Predicted ESOL', density=True)
plt.axvline(x=threshold, color='red', linestyle='dashed', linewidth=2, label=f'Median Split Threshold')
plt.title('Prediction Score Distribution Density')
plt.xlabel('Normalized ESOL Solubility')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('plot_1_score_distribution.png', dpi=300)
plt.close()

# PLOT 2: ROC Curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Tri-Modal ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Insoluble predicted as Soluble)')
plt.ylabel('True Positive Rate (Correctly predicted Soluble)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_2_roc_curve.png', dpi=300)
plt.close()

# PLOT 3: Precision-Recall Curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall (Fraction of High Solubility molecules found)')
plt.ylabel('Precision (Confidence when predicting High Solubility)')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_3_pr_curve.png', dpi=300)
plt.close()

# PLOT 4: Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Sol.', 'High Sol.'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix (Binarized Predictions)')
plt.tight_layout()
plt.savefig('plot_4_confusion_matrix.png', dpi=300)
plt.close()

print(">>> SUCCESS! 4 New Graphs generated successfully.")
print(">>> Saved as: plot_1_score_distribution.png, plot_2_roc_curve.png, plot_3_pr_curve.png, plot_4_confusion_matrix.png")