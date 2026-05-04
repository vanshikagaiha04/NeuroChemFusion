import torch
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import pandas as pd
import torch.nn as nn
import warnings
import os

warnings.filterwarnings("ignore")

# --- Architecture (Exactly same as V11) ---
class NeuroChemFusionTox11(nn.Module):
    def __init__(self, vocab_size, desc_dim, n_tasks=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = Mamba(d_model=128, d_state=64, d_conv=4, expand=2)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        self.struct_branch = nn.Sequential(
            nn.Linear(desc_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(128 * 3, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_tasks)
        )

    def forward(self, smiles_ids, descriptors):
        f_seq = self.mamba_branch(self.embedding(smiles_ids)).mean(dim=1)
        f_phys = self.phys_branch(descriptors)
        f_struct = self.struct_branch(descriptors)
        gates = self.gate_layer(torch.cat([f_seq, f_phys, f_struct], dim=-1))
        f_fusion = (gates[:, 0:1] * f_seq) + (gates[:, 1:2] * f_phys) + (gates[:, 2:3] * f_struct)
        return self.classifier(f_fusion)

# --- Data & Eval Setup ---
print(">>> Preparing Data for Deep Evaluation...", flush=True)
feat = dc.feat.RDKitDescriptors()
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer=feat, data_dir='./tox21_data_desc', reload=False)
train_ds, valid_ds, test_ds = datasets

tokenizer = dc.feat.BasicSmilesTokenizer()
unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
token_to_id['<PAD>'], token_to_id['<UNK>'] = 0, 1

def encode(smiles_list):
    encoded = [[token_to_id.get(t, 1) for t in tokenizer.tokenize(s)[:128]] + [0]*(128-len(tokenizer.tokenize(s)[:128])) for s in smiles_list]
    return torch.tensor(encoded, dtype=torch.long)

device = torch.device("cuda")
model = NeuroChemFusionTox11(len(token_to_id), test_ds.X.shape[1]).to(device)
model.load_state_dict(torch.load('best_tox21_v10.pt')) # Load your best weights
model.eval()

# --- Prediction Loop ---
print(">>> Running Predictions on Test Set...", flush=True)
s_test = encode(test_ds.ids).to(device)
d_test = torch.tensor(np.nan_to_num(test_ds.X.astype(np.float32)), dtype=torch.float32).to(device)
y_true = test_ds.y

with torch.no_grad():
    y_pred_logits = model(s_test, d_test)
    y_pred_prob = torch.sigmoid(y_pred_logits).cpu().numpy()
    y_pred_label = (y_pred_prob > 0.5).astype(int)

# --- Per-Task Metrics ---
task_results = []
for i, task_name in enumerate(tasks):
    valid_mask = ~np.isnan(y_true[:, i])
    t_true = y_true[valid_mask, i]
    t_prob = y_pred_prob[valid_mask, i]
    t_label = y_pred_label[valid_mask, i]
    
    if len(np.unique(t_true)) > 1:
        task_results.append({
            "Task": task_name,
            "ROC-AUC": roc_auc_score(t_true, t_prob),
            "PRC-AUC": average_precision_score(t_true, t_prob),
            "F1-Score": f1_score(t_true, t_label),
            "Precision": precision_score(t_true, t_label, zero_division=0),
            "Recall": recall_score(t_true, t_label, zero_division=0)
        })

df = pd.DataFrame(task_results)
df.to_csv("tox21_comprehensive_report.csv", index=False)

print("\n" + "="*80)
print(f"{'Task Name':<20} | {'ROC-AUC':<10} | {'PRC-AUC':<10} | {'F1-Score':<10}")
print("-" * 80)
for _, row in df.iterrows():
    print(f"{row['Task']:<20} | {row['ROC-AUC']:<10.4f} | {row['PRC-AUC']:<10.4f} | {row['F1-Score']:<10.4f}")
print("="*80)
print(f"OVERALL MEAN ROC-AUC: {df['ROC-AUC'].mean():.4f}")
print(f"OVERALL MEAN PRC-AUC: {df['PRC-AUC'].mean():.4f}")
print("="*80, flush=True)