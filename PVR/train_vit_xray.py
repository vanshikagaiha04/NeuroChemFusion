import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    f1_score, confusion_matrix, precision_score, recall_score
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import json, copy

# ==========================================
# 1. DATASET
# ==========================================
path      = "/home/kamakshi.rautela/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2"
data_dir  = os.path.join(path, 'chest_xray')
train_dir = os.path.join(data_dir, 'train')
test_dir  = os.path.join(data_dir, 'test')

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 50
PATIENCE   = 5
SEED       = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_train = datasets.ImageFolder(train_dir, transform=transform_train)
test_ds    = datasets.ImageFolder(test_dir,  transform=transform_test)

train_size = int(0.8 * len(full_train))
val_size   = len(full_train) - train_size
train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

labels_all  = [full_train.targets[i] for i in train_ds.indices]
n_normal    = labels_all.count(0)
n_pneumonia = labels_all.count(1)
total       = n_normal + n_pneumonia
w_normal    = total / (2 * n_normal)
w_pneumonia = total / (2 * n_pneumonia)
CLASS_WEIGHTS = torch.tensor([w_normal, w_pneumonia], dtype=torch.float)
print(f"Class weights → Normal: {w_normal:.3f}, Pneumonia: {w_pneumonia:.3f}")

# ==========================================
# 2. MODEL COMPONENTS
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ffn   = SwiGLU(embed_dim, ffn_hidden)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CustomAttentionModel(nn.Module):
    def __init__(self, n_layers=2, activation='swiglu', use_rms=True, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        embed_dim  = 128
        seq_len    = (IMG_SIZE // 8) ** 2
        ffn_hidden = embed_dim * 2

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        layers = []
        for _ in range(n_layers):
            if activation == 'swiglu' and use_rms:
                layers.append(AttentionBlock(embed_dim, 4, ffn_hidden, dropout))
            else:
                layers.append(self._plain_block(embed_dim, ffn_hidden, dropout, use_rms))
        self.encoder = nn.Sequential(*layers)

        self.norm = RMSNorm(embed_dim) if use_rms else nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    @staticmethod
    def _plain_block(embed_dim, ffn_hidden, dropout, use_rms):
        class PlainBlock(nn.Module):
            def __init__(self):
                super().__init__()
                Norm = RMSNorm if use_rms else nn.LayerNorm
                self.norm1 = Norm(embed_dim)
                self.norm2 = Norm(embed_dim)
                self.attn  = nn.MultiheadAttention(embed_dim, 4,
                                                   dropout=dropout, batch_first=True)
                self.ffn   = nn.Sequential(
                    nn.Linear(embed_dim, ffn_hidden),
                    nn.GELU(),
                    nn.Linear(ffn_hidden, embed_dim)
                )
                self.drop = nn.Dropout(dropout)
            def forward(self, x):
                n = self.norm1(x)
                a, _ = self.attn(n, n, n)
                x = x + self.drop(a)
                x = x + self.drop(self.ffn(self.norm2(x)))
                return x
        return PlainBlock()

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==========================================
# 3. TRAINING UTILITIES
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0., 0, 0
    all_preds, all_probs, all_labels = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)[:, 1]
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_probs), np.array(all_labels))


def run_training(model, tag, epochs, patience, class_weights, device,
                 train_loader, val_loader):
    cw        = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss":[], "val_loss":[],
               "train_acc":[], "val_acc":[], "stopped_epoch": epochs}
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    print(f"\n{'─'*55}\n Training: {tag}  ({model.count_params():,} params)\n{'─'*55}")

    for epoch in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"train loss={tl:.4f} acc={ta:.4f} | "
              f"val loss={vl:.4f} acc={va:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            patience_ctr  = 0
            best_state    = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  ⏹  Early stopping at epoch {epoch}")
                history["stopped_epoch"] = epoch
                break

    model.load_state_dict(best_state)
    return history


# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================

COLORS = {"train":"#2563EB", "val":"#DC2626", "roc":"#7C3AED", "f1":"#16A34A"}

def plot_curves(history, tag):
    ep = range(1, history["stopped_epoch"] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{tag} — Training Curves", fontweight="bold")

    ax1.plot(ep, history["train_loss"][:len(ep)], color=COLORS["train"], lw=2, label="Train Loss")
    ax1.plot(ep, history["val_loss"][:len(ep)],   color=COLORS["val"],   lw=2, label="Val Loss")
    ax1.axvline(history["stopped_epoch"], color="grey", ls=":", lw=1.2,
                label=f"Early stop (ep {history['stopped_epoch']})")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss"); ax1.legend()
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax2.plot(ep, [a*100 for a in history["train_acc"][:len(ep)]],
             color=COLORS["train"], lw=2, label="Train Acc")
    ax2.plot(ep, [a*100 for a in history["val_acc"][:len(ep)]],
             color=COLORS["val"],   lw=2, label="Val Acc")
    ax2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy"); ax2.legend()
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(f"{tag}_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {tag}_curves.png")


def plot_roc(y_true, y_scores, tag):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color=COLORS["roc"], lw=2.5, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],"k--", lw=1.2, label="Random (AUC=0.5)")
    plt.fill_between(fpr, tpr, alpha=0.07, color=COLORS["roc"])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {tag}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{tag}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {tag}_roc.png")
    return roc_auc


def plot_f1_curve(y_true, y_scores, tag):
    thresholds = np.linspace(0.01, 0.99, 100)
    f1s = [f1_score(y_true, (y_scores >= t).astype(int), zero_division=0)
           for t in thresholds]
    best_idx = int(np.argmax(f1s))
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s, color=COLORS["f1"], lw=2.5)
    plt.scatter(thresholds[best_idx], f1s[best_idx],
                color=COLORS["val"], s=100, zorder=5,
                label=f"Best F1={f1s[best_idx]:.4f} @ thr={thresholds[best_idx]:.2f}")
    plt.axvline(thresholds[best_idx], color=COLORS["val"], ls="--", lw=1.2)
    plt.xlabel("Threshold"); plt.ylabel("F1-Score")
    plt.title(f"F1-Score vs Threshold — {tag}", fontweight="bold")
    plt.legend(); plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.tight_layout()
    plt.savefig(f"{tag}_f1_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {tag}_f1_curve.png")


def plot_confusion_matrix(y_true, y_pred, tag):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Normal","Pneumonia"],
                yticklabels=["Normal","Pneumonia"],
                linewidths=0.5)
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.72, f"(n={cm[i,j]})",
                     ha="center", fontsize=9, color="grey")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix — {tag}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{tag}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {tag}_confusion.png")


def plot_ablation_bar(results):
    variants = list(results.keys())
    metrics  = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors   = ["#2563EB","#16A34A","#D97706","#7C3AED","#DC2626"]
    x = np.arange(len(variants)); width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [results[v].get(m, 0) for v in variants]
        bars = ax.bar(x + i*width - 2*width, vals, width,
                      label=m.upper().replace("_","-"), color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", fontsize=7.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylim([0.6, 1.05]); ax.set_ylabel("Score")
    ax.set_title("Ablation Study — All Variants", fontweight="bold")
    ax.legend(ncol=5, loc="lower right")
    plt.tight_layout()
    plt.savefig("ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ablation_comparison.png")


# ==========================================
# 5. ABLATION STUDY
# ==========================================

ABLATION_VARIANTS = {
    "V1-Baseline":    dict(n_layers=1, activation="gelu",   use_rms=False),
    "V2-RMSNorm":     dict(n_layers=1, activation="gelu",   use_rms=True),
    "V3-SwiGLU":      dict(n_layers=1, activation="swiglu", use_rms=False),
    "V4-RMS+SwiGLU":  dict(n_layers=1, activation="swiglu", use_rms=True),
    "V5-Proposed":    dict(n_layers=2, activation="swiglu", use_rms=True),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

ablation_results   = {}
ablation_histories = {}

for name, cfg in ABLATION_VARIANTS.items():
    model = CustomAttentionModel(**cfg).to(device)
    hist  = run_training(model, name, EPOCHS, PATIENCE,
                         CLASS_WEIGHTS, device, train_loader, val_loader)
    ablation_histories[name] = hist
    plot_curves(hist, name)

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    _, acc, preds, probs, labels = evaluate(model, test_loader, criterion, device)

    roc_auc   = auc(*roc_curve(labels, probs)[:2])
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels,    preds, average="macro", zero_division=0)
    f1        = f1_score(labels,        preds, average="macro", zero_division=0)

    ablation_results[name] = {
        "accuracy":  float(acc),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "roc_auc":   float(roc_auc),
        "stopped_epoch": hist["stopped_epoch"]
    }

    plot_roc(labels, probs, name)
    plot_f1_curve(labels, probs, name)
    plot_confusion_matrix(labels, preds, name)

plot_ablation_bar(ablation_results)

print("\nAblation Summary:")
print(f"{'Variant':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Ep':>5}")
print("─" * 60)
for k, v in ablation_results.items():
    print(f"{k:<20} {v['accuracy']:>7.4f} {v['precision']:>7.4f} "
          f"{v['recall']:>7.4f} {v['f1']:>7.4f} {v['roc_auc']:>7.4f} "
          f"{v['stopped_epoch']:>5d}")

with open("ablation_summary.json", "w") as f:
    json.dump(ablation_results, f, indent=2)


# ==========================================
# 6. FINAL MODEL EVALUATION (V5 Proposed)
# ==========================================
print("\n" + "═"*55)
print(" FINAL EVALUATION — V5-Proposed")
print("═"*55)

final_model = CustomAttentionModel(n_layers=2, activation="swiglu",
                                   use_rms=True).to(device)
final_hist  = run_training(final_model, "Final-ViT-XR", EPOCHS, PATIENCE,
                           CLASS_WEIGHTS, device, train_loader, val_loader)
plot_curves(final_hist, "Final-ViT-XR")

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
_, acc, preds, probs, labels = evaluate(final_model, test_loader, criterion, device)

print("\n--- Test Set Classification Report ---")
print(classification_report(labels, preds,
                             target_names=["Normal","Pneumonia"], digits=4))

roc_auc = auc(*roc_curve(labels, probs)[:2])
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  Stopped : epoch {final_hist['stopped_epoch']}")

plot_roc(labels, probs, "Final-ViT-XR")
plot_f1_curve(labels, probs, "Final-ViT-XR")
plot_confusion_matrix(labels, preds, "Final-ViT-XR")

final_results = {
    "accuracy":  float(acc),
    "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
    "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
    "f1":        float(f1_score(labels, preds, average="macro", zero_division=0)),
    "roc_auc":   float(roc_auc),
    "stopped_epoch": final_hist["stopped_epoch"],
    "total_params":  final_model.count_params()
}
with open("final_results.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\nFinal Results:", json.dumps(final_results, indent=2))
print("\nAll plots saved as PNG files in the current directory.")