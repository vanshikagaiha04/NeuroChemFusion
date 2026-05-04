import torch, os, numpy as np, deepchem as dc
from sklearn.metrics import roc_auc_score, r2_score
import warnings
from mamba_ssm import Mamba

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
    def forward(self, s, d, n): return self.classifier(self.backbone(s, d, n))

def run_final_best_eval():
    print(f">>> Running Final Device-Safe Evaluation on: {device}", flush=True)
    f_desc = dc.feat.RDKitDescriptors()
    
    eval_config = [
        ('bbbp', 'cls', './bbbp_data_desc', 'neurochem_bbbp_final.pt', 1, dc.molnet.load_bbbp),
        ('tox21', 'cls', './tox21_data_desc', 'neurochem_tox21_final.pt', 12, dc.molnet.load_tox21),
        ('esol', 'reg', './esol_data', 'neurochem_backbone_weights.pt', 1, dc.molnet.load_delaney),
        ('lipo', 'reg', './lipo_data', 'neurochem_backbone_weights.pt', 1, dc.molnet.load_lipo)
    ]

    print("\n" + "="*60)
    print(f"{'TASK':<12} | {'METRIC':<15} | {'SCORE':<10}")
    print("-" * 60)

    for name, t_type, d_dir, weight_file, n_tasks, loader in eval_config:
        if not os.path.exists(weight_file):
            print(f"{name.upper():<12} | ERROR: Weight file missing")
            continue
            
        try:
            _, datasets, _ = loader(featurizer=f_desc, data_dir=d_dir, reload=False)
            te_ds = datasets[2]
            X = np.nan_to_num(te_ds.X.astype(np.float32))
            if X.shape[1] < 217: X = np.pad(X, ((0,0),(0, 217-X.shape[1])))
            else: X = X[:, :217]
            y = np.nan_to_num(te_ds.y.astype(np.float32))

            backbone = SharedBackbone(52, 217).to(device)
            sd = torch.load(weight_file, map_location=device)
            
            if 'backbone' in weight_file: # Phase 1 Weights
                backbone.load_state_dict(sd)
                model = NeuroChemSpecialist(backbone, n_tasks).to(device)
                if t_type == 'reg':
                    # Matching the head used in Phase 1
                    model.classifier = torch.nn.Sequential(torch.nn.Linear(384, 1)).to(device)
            else: # Phase 2 Weights
                model = NeuroChemSpecialist(backbone, n_tasks).to(device)
                model.load_state_dict(sd)
            
            model.eval()
            with torch.no_grad():
                # ENSURING ALL TENSORS ARE ON THE SAME DEVICE
                s_t = torch.randint(0, 52, (len(X), 128)).to(device)
                d_t = torch.from_numpy(X).to(device)
                n_t = torch.randn(len(X), 20).to(device)
                
                out = model(s_t, d_t, n_t)
                preds = torch.sigmoid(out).cpu().numpy() if t_type == 'cls' else out.cpu().numpy()

            if t_type == 'cls':
                if name == 'tox21':
                    score = np.mean([roc_auc_score(y[:,i], preds[:,i]) for i in range(12) if len(np.unique(y[:,i])) > 1])
                else: score = roc_auc_score(y, preds)
                metric = "ROC-AUC"
            else:
                score = r2_score(y, preds)
                metric = "R2-Score"
            
            print(f"{name.upper():<12} | {metric:<15} | {score:.4f}")
        except Exception as e:
            print(f"{name.upper():<12} | EVAL ERROR: {str(e)[:30]}")
    print("="*60)

if __name__ == '__main__': run_final_best_eval()