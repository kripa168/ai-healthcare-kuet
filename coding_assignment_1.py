# =============================================================================
# Coding Assignment 1: Comparative Analysis of ML Classifiers for Medical Diagnosis
# Student  : Kripa Das  |  Roll: 2571557
# Dataset  : Breast Cancer Wisconsin (Diagnostic) - scikit-learn
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# Custom color palette (forest green theme)
COLORS = ["#2E7D32", "#66BB6A", "#A5D6A7"]

# =============================================================================
# PHASE A: DATA ENGINEERING (Pandas)
# =============================================================================
print("=" * 65)
print("  PHASE A: DATA ENGINEERING")
print("=" * 65)

# Load dataset and convert to structured DataFrame
raw = load_breast_cancer()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target   # 0 = Malignant, 1 = Benign

print(f"\n✔ Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"  Classes       : {list(raw.target_names)}  (0=Malignant, 1=Benign)")
print(f"  Benign        : {(df['target']==1).sum()}  |  Malignant: {(df['target']==0).sum()}")

# ── Missing Value Check ───────────────────────────────────────────────────────
null_sum = df.isnull().sum()
if null_sum.any():
    print(f"\n⚠ Missing values:\n{null_sum[null_sum > 0]}")
else:
    print("\n✔ No missing values found in any column.")

# ── Correlation Analysis — Top 5 Features ────────────────────────────────────
corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
top5 = corr.head(5).index.tolist()

print("\n✔ Top 5 features most correlated with target:")
for rank, feat in enumerate(top5, 1):
    print(f"   {rank}. {feat}  (|r| = {corr[feat]:.4f})")

# ── Feature Scaling with MinMaxScaler ────────────────────────────────────────
X = df[top5].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = MinMaxScaler()   # Scale to [0, 1] range
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✔ MinMaxScaler applied (values scaled to [0, 1])")
print(f"  Train : {X_train_sc.shape[0]} samples  |  Test : {X_test_sc.shape[0]} samples")

# =============================================================================
# PHASE B: MODEL IMPLEMENTATION (Scikit-Learn)
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE B: MODEL IMPLEMENTATION")
print("=" * 65)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=7),
    "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=7),
    "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "Accuracy": acc, "Precision": prec, "Recall": rec,
    }
    print(f"\n  [{name}]")
    print(f"   Accuracy : {acc:.4f}  |  Precision : {prec:.4f}  |  Recall : {rec:.4f}")
    report = classification_report(y_test, y_pred, target_names=raw.target_names)
    for line in report.splitlines():
        print("   " + line)

best_name = max(results, key=lambda k: results[k]["Accuracy"])
print(f"\n✔ Best Model: {best_name}  (Accuracy = {results[best_name]['Accuracy']:.4f})")

# =============================================================================
# PHASE C: VISUALIZATION (Matplotlib)
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE C: VISUALIZATION")
print("=" * 65)

# ── Plot 1: Model Comparison Bar Chart ───────────────────────────────────────
metric_names = ["Accuracy", "Precision", "Recall"]
model_names  = list(results.keys())
x     = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, (metric, color) in enumerate(zip(metric_names, COLORS)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Accuracy, Precision & Recall", fontsize=14, fontweight="bold")
ax.set_xticks(x + width); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.10); ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a1_plot1_model_comparison.png", dpi=150)
plt.close()
print("\n✔ Plot 1 saved: kripa_a1_plot1_model_comparison.png")

# ── Plot 2: Confusion Matrix (Best Model) ────────────────────────────────────
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=raw.target_names, yticklabels=raw.target_names,
            linewidths=0.5, linecolor="gray", ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — {best_name}\n(FP={cm[0,1]}  |  FN={cm[1,0]})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a1_plot2_confusion_matrix.png", dpi=150)
plt.close()
print("✔ Plot 2 saved: kripa_a1_plot2_confusion_matrix.png")

# ── Plot 3: ROC Curves (All Models) ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC = {auc(fpr,tpr):.3f})")
ax.plot([0,1],[0,1],"k--",lw=1.2,label="Random Chance")
ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10); ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a1_plot3_roc_curves.png", dpi=150)
plt.close()
print("✔ Plot 3 saved: kripa_a1_plot3_roc_curves.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
summary = pd.DataFrame({m:{k:f"{v:.4f}" for k,v in res.items() if k in metric_names} for m,res in results.items()}).T
summary.index.name = "Model"
print(f"\n{summary.to_string()}")
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {results[best_name]['Accuracy']:.4f}")
print(f"  Precision  : {results[best_name]['Precision']:.4f}")
print(f"  Recall     : {results[best_name]['Recall']:.4f}")
print("\n  ✔ Coding Assignment 1 (Kripa Das) — Complete!\n")
