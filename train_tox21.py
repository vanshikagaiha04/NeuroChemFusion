import os
import torch
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import warnings

# --- SETUP ---
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
warnings.filterwarnings("ignore")
device = torch.device("cuda")

# --- DATA PREP ---
def prepare_tox21_v12():
    print(">>> Loading Tox21 Data (V12)...", flush=True)
    featurizer = dc.feat.RDKitDescriptors()
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer=featurizer, data_dir='./tox21_data_desc', reload=False
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

    return train_ds, valid_ds, test_ds, encode_smiles, len(token_to_id), tasks

# --- ARCHITECTURE ---
class NeuroChemFusionTox12(nn.Module):
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
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, n_tasks)
        )

    def forward(self, smiles_ids, descriptors):
        f_seq = self.mamba_branch(self.embedding(smiles_ids)).mean(dim=1)
        f_phys = self.phys_branch(descriptors)
        f_struct = self.struct_branch(descriptors)
        gates = self.gate_layer(torch.cat([f_seq, f_phys, f_struct], dim=-1))
        f_fusion = (gates[:, 0:1] * f_seq) + (gates[:, 1:2] * f_phys) + (gates[:, 2:3] * f_struct)
        return self.classifier(f_fusion)

# --- UTILS ---
def to_tensors_v12(ds, encode_fn):
    s = encode_fn(ds.ids).to(device)
    raw_X = np.nan_to_num(ds.X.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    d = torch.tensor(raw_X, dtype=torch.float32).to(device)
    y = torch.tensor(ds.y, dtype=torch.float32).to(device)
    return s, d, y

def deep_evaluate(model, ds, encode_fn, tasks):
    model.eval()
    with torch.no_grad():
        s, d, y_true = to_tensors_v12(ds, encode_fn)
        logits = model(s, d)
        y_pred = torch.sigmoid(logits).cpu().numpy()
        y_true = y_true.cpu().numpy()
    
    metrics = []
    # Lower threshold (0.2) because Tox21 is imbalanced - helps with F1
    threshold = 0.2 
    
    for i, task in enumerate(tasks):
        valid = ~np.isnan(y_true[:, i])
        t_true = y_true[valid, i]
        t_pred = y_pred[valid, i]
        if len(np.unique(t_true)) > 1:
            auc = roc_auc_score(t_true, t_pred)
            prc = average_precision_score(t_true, t_pred)
            f1 = f1_score(t_true, (t_pred > threshold).astype(int), zero_division=0)
            metrics.append({"Task": task, "AUC": auc, "PRC": prc, "F1": f1})
    
    return metrics

# --- MAIN ---
train_ds, valid_ds, test_ds, encode_fn, vocab_sz, tasks = prepare_tox21_v12()
t_s, t_d, t_y = to_tensors_v12(train_ds, encode_fn)
v_s, v_d, v_y = to_tensors_v12(valid_ds, encode_fn)

model = NeuroChemFusionTox12(vocab_sz, t_d.shape[1]).to(device)

# AGGRESSIVE WEIGHTING: 25x weight to the positive class to kill F1=0
pos_weights = torch.tensor([25.0] * 12).to(device)
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)

print("\n>>> Training V12 (F1-Killer Mode)...", flush=True)
best_auc = 0

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    logits = model(t_s, t_d)
    raw_loss = criterion(logits, t_y)
    mask = ~torch.isnan(t_y)
    loss = raw_loss[mask].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 20 == 0:
        results = deep_evaluate(model, valid_ds, encode_fn, tasks)
        mean_auc = np.mean([r['AUC'] for r in results])
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {mean_auc:.4f}", flush=True)
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), 'best_tox21_v12.pt')

# --- FINAL DETAILED EVALUATION ---
print("\n>>> Running Final Detailed Evaluation...", flush=True)
model.load_state_dict(torch.load('best_tox21_v12.pt'))
final_results = deep_evaluate(model, test_ds, encode_fn, tasks)

print("-" * 65)
print(f"{'Task Name':<20} | {'ROC-AUC':<8} | {'PRC-AUC':<8} | {'F1-Score':<8}")
print("-" * 65)
for r in final_results:
    print(f"{r['Task']:<20} | {r['AUC']:.4f}   | {r['PRC']:.4f}   | {r['F1']:.4f}")
print("-" * 65)
print(f"MEAN TEST ROC-AUC: {np.mean([r['AUC'] for r in final_results]):.4f}")
print("-" * 65, flush=True)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_tox21_results(final_results):
    print("\n>>> Generating Tox21 Multi-Task Visuals...", flush=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(final_results)
    df.set_index('Task', inplace=True)

    # Setup the figure (Big enough for all 12 tasks)
    plt.figure(figsize=(20, 12))

    # 1. Bar Chart for ROC-AUC (Per Task)
    plt.subplot(2, 2, 1)
    df['AUC'].sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
    plt.axvline(x=0.7, color='red', linestyle='--', label='Baseline 0.7')
    plt.title('Tox21: ROC-AUC Per Task')
    plt.xlabel('ROC-AUC Score')
    plt.legend()

    # 2. Precision-Recall Curve Comparison (Top 5 Tasks)
    plt.subplot(2, 2, 2)
    df['PRC'].sort_values(ascending=False).head(5).plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title('Top 5 Tasks by PRC-AUC (Information Density)')
    plt.ylabel('PRC-AUC Score')
    plt.xticks(rotation=45)

    # 3. Heatmap of All Metrics
    plt.subplot(2, 2, 3)
    sns.heatmap(df[['AUC', 'PRC', 'F1']], annot=True, cmap='YlGnBu', cbar=False)
    plt.title('Tox21: Detailed Metrics Heatmap')

    # 4. Global Performance Summary
    plt.subplot(2, 2, 4)
    summary_metrics = df.mean()
    plt.pie(summary_metrics, labels=summary_metrics.index, autopct='%1.1f%%', 
            colors=['gold', 'lightcoral', 'lightskyblue'], startangle=140)
    plt.title(f'Mean Test Metrics\n(Global AUC: {df["AUC"].mean():.3f})')

    plt.tight_layout()
    plt.savefig('tox21_neurochem_report.png')
    print(">>> Final Report saved as 'tox21_neurochem_report.png'!", flush=True)

# Calling the function with your final_results
visualize_tox21_results(final_results)