import os, torch, warnings
import numpy as np
import deepchem as dc
from sklearn.metrics import roc_auc_score
from step2_biology_transfer import TransferBiologyModel, NeuroChemBackbone, ClassificationDataset, custom_collate_cls, encode_smiles, token_to_id
from torch.utils.data import DataLoader

os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n>>> Fixing BBBP Evaluation...", flush=True)

# 1. Load the exact same BBBP data
f_desc = dc.feat.RDKitDescriptors()
_, ds, _ = dc.molnet.load_bbbp(featurizer=f_desc, data_dir='./bbbp_data_desc', reload=False)
train_ds, valid_ds, test_ds = dc.splits.ScaffoldSplitter().train_valid_test_split(ds[0])
test_loader = DataLoader(ClassificationDataset(test_ds), batch_size=64, shuffle=False, collate_fn=custom_collate_cls)

# 2. Load the trained backbone and BBBP head
backbone = NeuroChemBackbone(len(token_to_id)+2, desc_dim=217)
model = TransferBiologyModel(backbone, n_tasks=1).to(device)

try:
    model.load_state_dict(torch.load('final_bbbp_model.pt'))
    print(">>> Loaded previously trained BBBP model!", flush=True)
except Exception as e:
    print(">>> Could not load model. Please run step2_biology_transfer.py again if needed.", flush=True)

# 3. Corrected Evaluation with MASKING
model.eval()
all_preds, all_trues, all_weights = [], [], []
with torch.no_grad():
    for s, d, n, e, ew, b_idx, y, w in test_loader:
        s, d, n, e, ew, b_idx = s.to(device), d.to(device), n.to(device), e.to(device), ew.to(device), b_idx.to(device)
        p = torch.sigmoid(model(s, d, n, e, ew, b_idx))
        p = torch.nan_to_num(p, nan=0.5) 
        
        all_preds.extend(p.cpu().numpy())
        all_trues.extend(y.numpy())
        all_weights.extend(w.numpy())

all_preds = np.array(all_preds)
all_trues = np.array(all_trues)
all_weights = np.array(all_weights)

# THE FIX: Mask out invalid weights just like Tox21
valid_mask = (all_weights > 0).flatten()

if valid_mask.sum() > 0 and len(np.unique(all_trues[valid_mask])) > 1:
    auc = roc_auc_score(all_trues[valid_mask], all_preds[valid_mask])
    print(f"\n[CORRECTED FINAL TEST] BBBP ROC-AUC = {auc:.4f} 🎯", flush=True)
else:
    print("\n[ERROR] Test set has only one class after masking. Try random splitting instead of scaffold splitting.", flush=True)