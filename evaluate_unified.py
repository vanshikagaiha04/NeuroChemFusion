import os, torch, warnings
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ARCHITECTURE ---
class NeuroChemFinal(nn.Module):
    def __init__(self, vocab_size, desc_dim, n_tasks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = Mamba(d_model=128, d_state=64, d_conv=4, expand=2)
        self.phys_branch = nn.Sequential(
            nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        self.graph_branch = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 128))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_tasks)
        )

    def forward(self, s, d, n):
        f_seq = self.mamba_branch(self.embedding(s)).mean(dim=1)
        f_phys = self.phys_branch(d)
        f_graph = self.graph_branch(n).mean(dim=0, keepdim=True).repeat(f_seq.size(0), 1)
        return self.classifier(torch.cat([f_seq, f_phys, f_graph], dim=-1))

def evaluate_task(task_type):
    print(f"\n>>> Evaluating Final Master Model on {task_type.upper()} Test Set...", flush=True)
    f_desc = dc.feat.RDKitDescriptors()
    
    if task_type == 'bbbp':
        _, datasets, _ = dc.molnet.load_bbbp(featurizer=f_desc, data_dir='./bbbp_data_desc', reload=False)
        n_tasks = 1
    else:
        _, datasets, _ = dc.molnet.load_tox21(featurizer=f_desc, data_dir='./tox21_data_desc', reload=False)
        n_tasks = 12

    # Get Test Data
    X_train = np.nan_to_num(datasets[0].X.astype(np.float32))
    scaler = StandardScaler().fit(X_train)
    X_test = scaler.transform(np.nan_to_num(datasets[2].X.astype(np.float32)))
    y_test = np.nan_to_num(datasets[2].y.astype(np.float32))
    w_test = np.nan_to_num(datasets[2].w.astype(np.float32))
    
    # Load Model
    model = NeuroChemFinal(52, X_test.shape[1], n_tasks).to(device)
    model.load_state_dict(torch.load(f'neurochem_{task_type}_MASTER.pt', map_location=device))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 32): 
            end = min(i + 32, len(X_test))
            batch_size = end - i
            s = torch.randint(0, 52, (batch_size, 128)).to(device) 
            d = torch.from_numpy(X_test[i:end]).to(device)
            n = torch.randn(batch_size, 20).to(device)
            logits = model(s, d, n)
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())

    test_preds = np.array(test_preds)

    # Calculate Scores
    if task_type == 'bbbp':
        auc_score = roc_auc_score(y_test, test_preds)
        print(f"[{task_type.upper()}] FINAL TEST ROC-AUC: {auc_score:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, test_preds)
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label=f'Unified BBBP (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title('BBBP Blood-Brain Barrier - ROC Curve')
        plt.legend()
        plt.savefig('unified_bbbp_roc.png', dpi=300)
        plt.close()
        
    elif task_type == 'tox21':
        task_aucs = [roc_auc_score(y_test[w_test[:, i] > 0, i], test_preds[w_test[:, i] > 0, i]) 
                     for i in range(12) if np.sum(w_test[:, i] > 0) > 0]
        print(f"[{task_type.upper()}] FINAL TEST AVERAGE ROC-AUC: {np.mean(task_aucs):.4f}")

if __name__ == '__main__':
    evaluate_task('bbbp')
    evaluate_task('tox21')
    print("\n>>> Done! Check 'unified_bbbp_roc.png' for the graph.")
