import os, torch, warnings
import torch.nn as nn
import numpy as np
import deepchem as dc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                             precision_score, recall_score, average_precision_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import rankdata

# --- SYSTEM SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODULAR DATA PREP (Future-Ready) ---
def prepare_bbbp_universal():
    print(">>> Loading BBBP (Universal Tri-modal Interface)...", flush=True)
    f_desc = dc.feat.RDKitDescriptors()
    _, datasets, _ = dc.molnet.load_bbbp(featurizer=f_desc, splitter='scaffold', data_dir='./bbbp_data_desc', reload=False)
    tr_ds, va_ds, te_ds = datasets
    
    scaler = StandardScaler()
    sel = VarianceThreshold(0.01)
    
    def process_features(ds, is_train=False):
        # 1. Physics Branch (Descriptors)
        raw_x = np.nan_to_num(ds.X.astype(np.float32), nan=0.0)
        desc = sel.fit_transform(scaler.fit_transform(raw_x)) if is_train else sel.transform(scaler.transform(raw_x))
        
        # 2. Structural Branch (Morgan Fingerprints)
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        fps = []
        for smi in ds.ids:
            mol = Chem.MolFromSmiles(str(smi))
            arr = np.zeros(2048, dtype=np.float32)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        
        # Combined interface for Fusion
        return np.concatenate([np.array(fps), desc], axis=1).astype(np.float32), ds.y.ravel()

    X_tr, y_tr = process_features(tr_ds, is_train=True)
    X_va, y_va = process_features(va_ds)
    X_te, y_te = process_features(te_ds)
    return X_tr, y_tr, X_va, y_va, X_te, y_te

# --- ARCHITECTURE (NeuroChem Standard MLP) ---
class NeuroChemMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x.float())

# --- FUSION ENGINE ---
def run_v34_fusion():
    X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_bbbp_universal()
    
    # Model A: Random Forest (Physics/Statistical Logic)
    rf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=3, class_weight='balanced', n_jobs=-1).fit(X_tr, y_tr)
    
    # Model B: Neural Network (Structural Context)
    Xt, yt = torch.from_numpy(X_tr).to(device), torch.from_numpy(y_tr).unsqueeze(1).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    model = NeuroChemMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.2)
    
    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); nn.BCEWithLogitsLoss()(model(xb), yb).backward(); opt.step()

    # --- RANK-BASED CALIBRATION (Validation Set) ---
    model.eval()
    with torch.no_grad():
        p_nn_va = torch.sigmoid(model(torch.from_numpy(X_va).to(device))).cpu().numpy().ravel()
    p_rf_va = rf.predict_proba(X_va)[:, 1]
    
    # Rank Pooling: Normalizing scores from different models
    # This is the most robust way to fuse NN and RF
    p_val_ens = (rankdata(p_rf_va) + rankdata(p_nn_va)) / (2 * len(y_va))
    
    # Optimize Threshold for F1-Score (The Harmonic Mean of Precision & Recall)
    thresholds = np.linspace(0.2, 0.8, 100)
    best_thr = 0.5
    best_f1 = 0
    
    for t in thresholds:
        f1 = f1_score(y_va, (p_val_ens > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
            
    # --- TEST INFERENCE ---
    with torch.no_grad():
        p_nn_te = torch.sigmoid(model(torch.from_numpy(X_te).to(device))).cpu().numpy().ravel()
    p_rf_te = rf.predict_proba(X_te)[:, 1]
    
    # Apply Rank-Based Fusion to Test Set
    p_final = (rankdata(p_rf_te) + rankdata(p_nn_te)) / (2 * len(y_te))
    y_pred = (p_final > best_thr).astype(int)

    # --- REPORTING ---
    print(f"\n{'-'*45}\n      NEUROCHEM V34 (UNIVERSAL ROBUST FUSION)\n{'-'*45}")
    print(f"Calibration Method : Rank-Pooling | Threshold: {best_thr:.2f}")
    print(f"ROC-AUC   : {roc_auc_score(y_te, p_final):.4f}")
    print(f"Accuracy  : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision : {precision_score(y_te, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_te, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_te, y_pred):.4f}")
    print(f"{'-'*45}\n")

    print("\n>>> Generating Visuals...", flush=True)
    plt.figure(figsize=(20, 5))

    # 1. Confusion Matrix
    plt.subplot(1, 4, 1)
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 2. ROC Curve
    plt.subplot(1, 4, 2)
    fpr, tpr, _ = roc_curve(y_te, p_final)
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_te, p_final):.3f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()

    # 3. Precision-Recall Curve
    plt.subplot(1, 4, 3)
    prec, rec, _ = precision_recall_curve(y_te, p_final)
    plt.plot(rec, prec, label=f'PRC: {average_precision_score(y_te, p_final):.3f}', color='green', lw=2)
    plt.title('PR-Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # 4. Probability Distribution
    plt.subplot(1, 4, 4)
    sns.histplot(p_final[y_te == 1], color="green", label="Penetrator", kde=True, stat="density", alpha=0.5)
    sns.histplot(p_final[y_te == 0], color="red", label="Non-Penetrator", kde=True, stat="density", alpha=0.5)
    plt.axvline(best_thr, color='black', linestyle='--', label=f'Thr: {best_thr}')
    plt.title('Score Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('neurochem_final_report.png')

if __name__ == '__main__':
    run_v34_fusion()