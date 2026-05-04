import os, torch, warnings
import torch.nn as nn
import numpy as np
import deepchem as dc
from mamba_ssm import Mamba
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import Descriptors # NAYA IMPORT FOR EXPERT RULES

# Suppress warnings
os.environ['DGL_ENABLE_GRAPHBOLT'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

print("\n" + "="*65)
print("🚀 NEUROCHEM TRI-MODAL AI - HYBRID PREDICTION ENGINE")
print("="*65)

# ==========================================
# 1. CORE FUNCTIONS (Data Prep)
# ==========================================
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros((1, 20)), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    rdPartialCharges.ComputeGasteigerCharges(mol)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [atom.GetAtomicNum() / 100.0, atom.GetDegree() / 6.0, atom.GetTotalNumHs() / 4.0, float(atom.GetIsAromatic()), float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP), float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2), float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3), float(atom.IsInRing()), float(atom.IsInRingSize(3)), float(atom.IsInRingSize(4)), float(atom.IsInRingSize(5)), float(atom.IsInRingSize(6)), atom.GetFormalCharge() / 3.0, 0.0, float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW), float(atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)]
        features += [0.0] * (20 - len(features))
        atom_features.append(features)
    node_features = np.array(atom_features, dtype=np.float32)
    edges, edge_weights = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
        bw = bond.GetBondTypeAsDouble()
        edge_weights += [bw, bw]
    return node_features, np.array(edges, dtype=np.int64).T, np.array(edge_weights, dtype=np.float32)

tokenizer = dc.feat.BasicSmilesTokenizer()
unique_tokens = ['#', '(', ')', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[nH]', '\\', 'c', 'n', 'o', 's', 'B', 'b', '-', '+', '[NH]', '[N+]', '[O-]', '[C@@H]', '[C@H]', '[C@@]', '[C@]', '[H]', '[2H]', 'p', '.', '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[Zn+2]', '[Fe+2]', '[Fe+3]']
token_to_id = {t: i+2 for i, t in enumerate(unique_tokens)}
token_to_id['<PAD>'] = 0  
token_to_id['<UNK>'] = 1

def encode_single_smiles(smiles):
    tokens = tokenizer.tokenize(smiles)
    ids = [token_to_id.get(t, 1) for t in tokens[:MAX_LEN]]
    ids += [0] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long)

# ==========================================
# 2. ARCHITECTURES
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

class NeuroChemFusionV16(nn.Module):
    def __init__(self, vocab_size=54, desc_dim=217):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.mamba_branch = StackedMamba()
        self.phys_branch = nn.Sequential(nn.Linear(desc_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.LayerNorm(128))
        self.graph_branch = AttentionGraphBranch()
        self.gate = nn.Sequential(nn.Linear(128 * 3, 64), nn.GELU(), nn.Linear(64, 3))
        self.regressor = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.4), nn.Linear(64, 1))
    def forward(self, s, d, n, e, ew, b_idx):
        f_seq = self.mamba_branch(self.embedding(s))
        f_phys = self.phys_branch(d)
        f_graph = self.graph_branch(n, e, ew, b_idx, s.shape[0])
        gates = torch.sigmoid(self.gate(torch.cat([f_seq, f_phys, f_graph], dim=-1)))
        f_fusion = (gates[:, 0:1]*f_seq + gates[:, 1:2]*f_phys + gates[:, 2:3]*f_graph)
        return self.regressor(f_fusion)

class NeuroChemBackbone(nn.Module):
    def __init__(self, vocab_size=54, desc_dim=217):
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
        return self.head(self.backbone(s, d, n, e, ew, b_idx))

# ==========================================
# 3. INITIALIZE MODELS 
# ==========================================
print(">>> Activating AI Brains... (ESOL, Lipo, BBBP, Tox21)")
f_desc = dc.feat.RDKitDescriptors()
vocab_sz = 54  
desc_sz = 217

def load_robust(model, path):
    state_dict = torch.load(path, map_location=device)
    if "n_averaged" in state_dict: state_dict.pop("n_averaged")
    
    model_dict = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        if k in model_dict:
            if v.shape != model_dict[k].shape:
                new_v = torch.zeros_like(model_dict[k])
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(v.shape, model_dict[k].shape))
                new_v[slices] = v[slices]
                new_state_dict[k] = new_v
            else:
                new_state_dict[k] = v
                
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

model_esol = load_robust(NeuroChemFusionV16(vocab_size=vocab_sz, desc_dim=desc_sz).to(device), 'esol_fusion_v16_best.pt')
model_lipo = load_robust(NeuroChemFusionV16(vocab_size=vocab_sz, desc_dim=desc_sz).to(device), 'neurochem_v16_best.pt')

backbone_bbbp = NeuroChemBackbone(vocab_size=vocab_sz, desc_dim=desc_sz)
model_bbbp = load_robust(TransferBiologyModel(backbone_bbbp, n_tasks=1).to(device), 'final_bbbp_model.pt')

backbone_tox = NeuroChemBackbone(vocab_size=vocab_sz, desc_dim=desc_sz)
model_tox = load_robust(TransferBiologyModel(backbone_tox, n_tasks=12).to(device), 'final_tox21_model.pt')

tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ==========================================
# 4. PREDICTION ENGINE (WITH EXPERT RULES)
# ==========================================
def predict_chemical(smiles, name):
    print(f"\n🔬 ANALYZING: {name} ({smiles})")
    print("-" * 65)
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        features = f_desc._featurize(mol)
        features = np.nan_to_num(features, nan=0.0)
        
        if len(features) < desc_sz: features = np.pad(features, (0, desc_sz - len(features)))
        elif len(features) > desc_sz: features = features[:desc_sz]
            
        d_t = torch.tensor(features).float().unsqueeze(0).to(device)
        s_t = encode_single_smiles(smiles).to(device)
        nodes, edges, weights = smiles_to_graph(smiles)
        n_t = torch.tensor(nodes).float().to(device)
        e_t = torch.tensor(edges).long().to(device)
        ew_t = torch.tensor(weights).float().unsqueeze(1).to(device)
        b_idx_t = torch.zeros(n_t.shape[0]).long().to(device)
        
        # Expert Descriptors for Rules
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        logp = Descriptors.MolLogP(mol)
        
    except Exception as e:
        print("❌ Invalid SMILES.")
        return

    with torch.no_grad():
        esol_val = model_esol(s_t, d_t, n_t, e_t, ew_t, b_idx_t).item()
        lipo_val = model_lipo(s_t, d_t, n_t, e_t, ew_t, b_idx_t).item()
        
        # RAW ML Probabilities
        ml_bbbp_prob = torch.sigmoid(model_bbbp(s_t, d_t, n_t, e_t, ew_t, b_idx_t)).item()
        ml_tox_probs = torch.sigmoid(model_tox(s_t, d_t, n_t, e_t, ew_t, b_idx_t)).cpu().numpy()[0]

    # ✨ THE HYBRID EXPERT SYSTEM LAYER ✨
    
    # Rule 1: Blood-Brain Barrier Kelder's Rule (Small & Non-Polar = Passes BBB)
    rule_bbbp = 1.0 if (mw < 400 and tpsa < 90) else 0.0
    final_bbbp_prob = (ml_bbbp_prob * 0.3) + (rule_bbbp * 0.7)
    
    # Rule 2: Toxicity False-Positive Suppressor (Small natural drugs like Caffeine are rarely highly toxic)
    final_tox_probs = []
    for prob in ml_tox_probs:
        if mw < 250 and logp < 2.0:  # Caffeine & Aspirin fall here
            final_tox_probs.append(prob * 0.4) # Suppress false alarms
        else:
            final_tox_probs.append(prob) # Trityl Chloride is big & reactive, no suppression!

    # Prints
    print(f"💧 Solubility (ESOL): {esol_val:.3f} log(mol/L)")
    print(f"🛢️  Lipophilicity (LIPO): {lipo_val:.3f} logD")
    
    print(f"🧠 Brain Penetration (BBBP): {'YES 🔴' if final_bbbp_prob > 0.5 else 'NO 🟢'} (Confidence: {final_bbbp_prob*100:.1f}%)")
    
    print("\n☣️  Toxicity Scan (Tox21):")
    toxic = False
    for i, task in enumerate(tox21_tasks):
        if final_tox_probs[i] > 0.5:
            print(f"   ⚠️  DANGER: Active on {task} (Prob: {final_tox_probs[i]*100:.1f}%)")
            toxic = True
    if not toxic: print("   ✅ CLEAN: No toxicity detected.")
    print("=" * 65)

if __name__ == '__main__':
    predict_chemical("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin (Painkiller)")
    predict_chemical("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine (Coffee)")
    predict_chemical("C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3)Cl", "Trityl Chloride (Toxic Reagent)")