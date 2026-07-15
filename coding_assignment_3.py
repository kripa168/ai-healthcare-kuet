# =============================================================================
# Coding Assignment 3: Image-Based Cancer Diagnosis Using CNNs
# Student  : Kripa Das  |  Roll: 2571557
# Dataset  : Histopathologic Cancer Detection (synthetic structure)
#            Replace data generation block with real Kaggle / BUSI dataset.
# Framework: PyTorch
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import os, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 99
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*65}")
print(f"  Coding Assignment 3: CNN-Based Histopathologic Cancer Diagnosis")
print(f"  Student: Kripa Das  |  Roll: 2571557")
print(f"{'='*65}")
print(f"  Device : {DEVICE}")

# =============================================================================
# SYNTHETIC DATASET GENERATION
# Replace this block with your real Kaggle / BUSI dataset loader.
# Expected folder structure:
#   root/benign/     *.png or *.jpg
#   root/malignant/  *.png or *.jpg
# =============================================================================
DATASET_DIR  = "/tmp/kripa_cancer_dataset"
CLASSES      = ["benign", "malignant"]
IMG_SIZE     = 96
N_PER_CLASS  = 350

def generate_histopath_dataset(out_dir, n_per_class, img_size):
    """
    Synthetic histopathology-style images:
    Benign     → tiled uniform texture (simulating normal tissue).
    Malignant  → irregular dark clusters with random scatter (simulating tumor cells).
    """
    os.makedirs(out_dir, exist_ok=True)
    for label in CLASSES:
        cls_dir = os.path.join(out_dir, label)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n_per_class):
            # Base tissue color
            base = 210 if label == "benign" else 170
            arr  = np.full((img_size, img_size, 3), base, dtype=np.uint8)
            arr[:,:,0] = np.clip(arr[:,:,0] + np.random.randint(-20,20,(img_size,img_size)), 0, 255)
            arr[:,:,2] = np.clip(arr[:,:,2] - 30, 0, 255)  # pinkish hue

            if label == "malignant":
                # Scatter dark irregular clusters
                for _ in range(random.randint(5, 12)):
                    cx, cy = random.randint(10, img_size-10), random.randint(10, img_size-10)
                    r = random.randint(4, 12)
                    y_g, x_g = np.ogrid[:img_size, :img_size]
                    mask = (x_g-cx)**2 + (y_g-cy)**2 <= r**2
                    arr[mask] = [random.randint(40,90), random.randint(20,60), random.randint(20,60)]
            else:
                # Regular grid pattern (normal glands)
                for gx in range(0, img_size, 16):
                    arr[gx:gx+1, :] = [180, 160, 175]
                for gy in range(0, img_size, 16):
                    arr[:, gy:gy+1] = [180, 160, 175]

            img = Image.fromarray(arr, mode="RGB")
            img.save(os.path.join(cls_dir, f"{label}_{i:04d}.png"))

if not os.path.exists(DATASET_DIR):
    print("\n  Generating synthetic histopathology dataset...")
    generate_histopath_dataset(DATASET_DIR, N_PER_CLASS, IMG_SIZE)
    print(f"  ✔ Dataset created: {N_PER_CLASS*2} images ({N_PER_CLASS} per class)")
else:
    print(f"\n  ✔ Dataset found at {DATASET_DIR}")

# =============================================================================
# PHASE A: COMPUTER VISION DATA PIPELINE
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE A: COMPUTER VISION DATA PIPELINE")
print(f"{'='*65}")

class HistoDataset(Dataset):
    """
    Custom Dataset for histopathology image classification.
    Loads images from a root directory with class subfolders.
    """
    def __init__(self, root_dir, transform=None):
        self.samples = []; self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root_dir)))}
        for cls, idx in self.class_to_idx.items():
            for fname in os.listdir(os.path.join(root_dir, cls)):
                if fname.lower().endswith((".png",".jpg",".jpeg")):
                    self.samples.append((os.path.join(root_dir, cls, fname), idx))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i][0]).convert("RGB")
        return (self.transform(img) if self.transform else img), self.samples[i][1]

MEAN = [0.7, 0.55, 0.65]   # Approximate histopathology RGB mean
STD  = [0.15, 0.15, 0.15]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

full  = HistoDataset(DATASET_DIR, transform=train_tf)
total = len(full)
n_tr  = int(0.70 * total)
n_va  = int(0.15 * total)
n_te  = total - n_tr - n_va

tr_set, va_set, te_set = random_split(full, [n_tr, n_va, n_te],
                                       generator=torch.Generator().manual_seed(SEED))
va_set.dataset = HistoDataset(DATASET_DIR, transform=eval_tf)
te_set.dataset = HistoDataset(DATASET_DIR, transform=eval_tf)

tr_loader = DataLoader(tr_set, batch_size=32, shuffle=True,  num_workers=0)
va_loader = DataLoader(va_set, batch_size=32, shuffle=False, num_workers=0)
te_loader = DataLoader(te_set, batch_size=32, shuffle=False, num_workers=0)

print(f"\n  Class mapping  : {full.class_to_idx}")
print(f"  Total images   : {total}")
print(f"  Train / Val / Test : {n_tr} / {n_va} / {n_te}  (70/15/15%)")
print(f"  Image size     : {IMG_SIZE}×{IMG_SIZE} RGB")
print(f"  Augmentations  : HFlip, VFlip, Rotation(±20°), ColorJitter")

# =============================================================================
# PHASE B: CNN ARCHITECTURE & TRAINING
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE B: CNN ARCHITECTURE & TRAINING")
print(f"{'='*65}")

class HistoCNN(nn.Module):
    """
    Deep CNN for Histopathologic Cancer Detection.

    Architecture:
        3 Conv Blocks  → Conv2d → BatchNorm → ReLU → MaxPool
        Dropout        → p = 0.45
        FC Head        → Linear(512) → ReLU → Dropout → Linear(1) → Sigmoid
    """
    def __init__(self, dropout_p=0.45):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2, 2))          # 96 → 48
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2, 2))          # 48 → 24
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True), nn.MaxPool2d(2, 2))          # 24 → 12
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.block3(self.block2(self.block1(x))))


model     = HistoCNN(dropout_p=0.45).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=8e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
EPOCHS    = 25

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Model     : HistoCNN")
print(f"  Layers    : 3 Conv Blocks (Conv→BN→ReLU→MaxPool) + FC(512) Head")
print(f"  Dropout   : p = 0.45")
print(f"  Loss      : BCELoss")
print(f"  Optimizer : Adam (lr=8e-4) + CosineAnnealingLR")
print(f"  Params    : {total_params:,}")
print(f"\n  Training for {EPOCHS} epochs...")
print(f"  {'Epoch':>5} | {'TLoss':>8} | {'TAcc':>7} | {'VLoss':>8} | {'VAcc':>7}")
print(f"  {'-'*48}")

tr_losses, va_losses, tr_accs, va_accs = [], [], [], []

for epoch in range(1, EPOCHS+1):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for imgs, labs in tr_loader:
        imgs = imgs.to(DEVICE); labs = labs.float().unsqueeze(1).to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs); loss = criterion(preds, labs)
        loss.backward(); optimizer.step()
        tl += loss.item()*imgs.size(0)
        tc += ((preds>=0.5).float()==labs).sum().item(); tt += imgs.size(0)
    scheduler.step()

    model.eval(); vl, vc, vt = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labs in va_loader:
            imgs = imgs.to(DEVICE); labs = labs.float().unsqueeze(1).to(DEVICE)
            preds = model(imgs); loss = criterion(preds, labs)
            vl += loss.item()*imgs.size(0)
            vc += ((preds>=0.5).float()==labs).sum().item(); vt += imgs.size(0)

    tr_losses.append(tl/tt); va_losses.append(vl/vt)
    tr_accs.append(tc/tt);   va_accs.append(vc/vt)
    print(f"  {epoch:>5} | {tl/tt:>8.4f} | {tc/tt:>7.4f} | {vl/vt:>8.4f} | {vc/vt:>7.4f}")

print(f"\n  ✔ Training complete.")

# ── Test Evaluation ───────────────────────────────────────────────────────────
model.eval(); al, ap, apr = [], [], []
with torch.no_grad():
    for imgs, labs in te_loader:
        probs = model(imgs.to(DEVICE)).squeeze(1).cpu().numpy()
        apr.extend(probs); ap.extend((probs>=0.5).astype(int)); al.extend(labs.numpy())
al = np.array(al); ap = np.array(ap); apr = np.array(apr)
test_acc = accuracy_score(al, ap)
print(f"\n  Test Accuracy : {test_acc:.4f}")
report = classification_report(al, ap, target_names=["benign","malignant"])
for line in report.splitlines(): print("  " + line)

# =============================================================================
# PHASE C: VISUALIZATION
# =============================================================================
print(f"\n{'='*65}")
print(f"  PHASE C: VISUALIZATION")
print(f"{'='*65}")

er = range(1, EPOCHS+1)

# ── Plot 1: Learning Curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(er, tr_losses, "o-", color="#1B5E20", lw=2, label="Train Loss")
ax1.plot(er, va_losses, "s--", color="#66BB6A", lw=2, label="Val Loss")
ax1.set_xlabel("Epoch",fontsize=12); ax1.set_ylabel("BCE Loss",fontsize=12)
ax1.set_title("Training vs Validation Loss",fontsize=13,fontweight="bold")
ax1.legend(fontsize=10); ax1.grid(linestyle="--",alpha=0.5)

ax2.plot(er, tr_accs, "o-", color="#2E7D32", lw=2, label="Train Accuracy")
ax2.plot(er, va_accs, "s--", color="#A5D6A7", lw=2, label="Val Accuracy")
ax2.set_xlabel("Epoch",fontsize=12); ax2.set_ylabel("Accuracy",fontsize=12)
ax2.set_title("Training vs Validation Accuracy",fontsize=13,fontweight="bold")
ax2.legend(fontsize=10); ax2.grid(linestyle="--",alpha=0.5); ax2.set_ylim(0,1.05)
plt.suptitle("HistoCNN Learning Curves — Histopathologic Cancer Detection",
             fontsize=14,fontweight="bold",y=1.02)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a3_plot1_learning_curves.png",dpi=150,bbox_inches="tight")
plt.close()
print("\n  ✔ Plot 1 saved: kripa_a3_plot1_learning_curves.png")

# ── Plot 2: ROC Curve ────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(al, apr); roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7,6))
ax.plot(fpr, tpr, color="#2E7D32", lw=2.5, label=f"HistoCNN  (AUC = {roc_auc:.4f})")
ax.fill_between(fpr, tpr, alpha=0.08, color="#2E7D32")
ax.plot([0,1],[0,1],"k--",lw=1.2,label="Random Chance (AUC = 0.5)")
ax.set_xlabel("False Positive Rate",fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)
ax.set_title("ROC Curve — HistoCNN (Test Set)",fontsize=13,fontweight="bold")
ax.legend(loc="lower right",fontsize=11); ax.grid(linestyle="--",alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a3_plot2_roc_curve.png",dpi=150)
plt.close()
print("  ✔ Plot 2 saved: kripa_a3_plot2_roc_curve.png")

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(al, ap); tn, fp, fn, tp2 = cm.ravel()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Benign","Malignant"], yticklabels=["Benign","Malignant"],
            linewidths=0.5, linecolor="gray", ax=ax, annot_kws={"size":16,"weight":"bold"})
ax.set_xlabel("Predicted Label",fontsize=12); ax.set_ylabel("True Label",fontsize=12)
ax.set_title(f"Confusion Matrix — HistoCNN (Test Set)\nTP={tp2}  TN={tn}  FP={fp}  FN={fn}",
             fontsize=12,fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a3_plot3_confusion_matrix.png",dpi=150)
plt.close()
print("  ✔ Plot 3 saved: kripa_a3_plot3_confusion_matrix.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY — Kripa Das  |  Roll: 2571557")
print(f"{'='*65}")
print(f"\n  Architecture  : HistoCNN (3 Conv Blocks + FC(512) Head)")
print(f"  Parameters    : {total_params:,}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Test Accuracy : {test_acc:.4f}")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(f"  TP={tp2}  TN={tn}  FP={fp}  FN={fn}")
print(f"\n  ✔ Coding Assignment 3 (Kripa Das) — Complete!\n")
