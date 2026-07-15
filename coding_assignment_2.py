# =============================================================================
# Coding Assignment 2: Advanced Ensemble Learning and Evaluation for Cancer Prediction
# Student  : Kripa Das  |  Roll: 2571557
# Dataset  : Breast Cancer Wisconsin (Diagnostic) - scikit-learn
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings("ignore")

# Forest green color palette
COLORS = ["#1B5E20", "#388E3C", "#81C784"]

# =============================================================================
# PHASE A: DATA ENGINEERING & FEATURE SELECTION
# =============================================================================
print("=" * 65)
print("  PHASE A: DATA ENGINEERING & FEATURE SELECTION")
print("=" * 65)

raw = load_breast_cancer()
df  = pd.DataFrame(raw.data, columns=raw.feature_names)
df["target"] = raw.target

print(f"\n✔ Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"  Class distribution:\n{df['target'].value_counts().rename({0:'Malignant',1:'Benign'}).to_string()}")

# ── Data Integrity Check ──────────────────────────────────────────────────────
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"\n⚠ Null values found:\n{null_counts[null_counts > 0]}")
else:
    print("\n✔ Data integrity check: Zero null or missing values detected.")

# ── Feature Selection: Top 5 by Variance Threshold + Correlation ─────────────
# Step 1: Remove low-variance features
var_series = df.drop("target", axis=1).var().sort_values(ascending=False)
high_var    = var_series.head(15).index.tolist()

# Step 2: Among high-variance features, pick top 5 correlated with target
corr = df[high_var + ["target"]].corr()["target"].drop("target").abs().sort_values(ascending=False)
top5 = corr.head(5).index.tolist()

print("\n✔ Top 5 features (variance threshold + correlation):")
for i, feat in enumerate(top5, 1):
    print(f"   {i}. {feat}  (|r| = {corr[feat]:.4f})")

# ── StandardScaler ────────────────────────────────────────────────────────────
X = df[top5].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n✔ StandardScaler applied (zero mean, unit variance)")
print(f"  Train : {X_train_sc.shape[0]}  |  Test : {X_test_sc.shape[0]}")

# =============================================================================
# PHASE B: MODEL IMPLEMENTATION & HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE B: MODEL IMPLEMENTATION & TUNING")
print("=" * 65)

# ── Model 1: Decision Tree (Baseline) ────────────────────────────────────────
print("\n  [1] Decision Tree Classifier (Baseline)")
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=7),
    {"max_depth":[3,5,7,None], "min_samples_split":[2,5,10], "criterion":["gini","entropy"]},
    cv=5, scoring="accuracy", n_jobs=-1
)
dt_grid.fit(X_train_sc, y_train)
dt_best = dt_grid.best_estimator_
print(f"   Best params : {dt_grid.best_params_}")
print(f"   CV Accuracy : {dt_grid.best_score_:.4f}")

# ── Model 2: AdaBoost (Advanced Boosting Ensemble) ───────────────────────────
print("\n  [2] AdaBoost Classifier (Advanced Boosting Ensemble)")
ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=7),
    {"n_estimators":[50,100,200], "learning_rate":[0.5,1.0,1.5]},
    cv=5, scoring="accuracy", n_jobs=-1
)
ada_grid.fit(X_train_sc, y_train)
ada_best = ada_grid.best_estimator_
print(f"   Best params : {ada_grid.best_params_}")
print(f"   CV Accuracy : {ada_grid.best_score_:.4f}")

# ── Model 3: SVM RBF Kernel ──────────────────────────────────────────────────
print("\n  [3] Support Vector Machine — RBF Kernel")
svm_grid = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=7),
    {"C":[0.1,1,10,100], "gamma":["scale","auto",0.01,0.1]},
    cv=5, scoring="accuracy", n_jobs=-1
)
svm_grid.fit(X_train_sc, y_train)
svm_best = svm_grid.best_estimator_
print(f"   Best params : {svm_grid.best_params_}")
print(f"   CV Accuracy : {svm_grid.best_score_:.4f}")

# ── Test Set Evaluation ───────────────────────────────────────────────────────
models = {"Decision Tree": dt_best, "AdaBoost": ada_best, "SVM (RBF)": svm_best}
results = {}
print("\n  ── Test Set Evaluation ─────────────────────────────────────")
for name, model in models.items():
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model":model,"y_pred":y_pred,"y_proba":y_proba,
                     "Accuracy":acc,"F1-Score":f1,"ROC-AUC":roc_auc}
    print(f"\n  [{name}]")
    print(f"   Accuracy:{acc:.4f}  F1:{f1:.4f}  ROC-AUC:{roc_auc:.4f}")
    report = classification_report(y_test, y_pred, target_names=raw.target_names)
    for line in report.splitlines():
        print("   " + line)

best_name = max(results, key=lambda k: results[k]["ROC-AUC"])
print(f"\n✔ Best Model (ROC-AUC): {best_name}  ({results[best_name]['ROC-AUC']:.4f})")

# =============================================================================
# PHASE C: VISUALIZATION & ADVANCED EVALUATION
# =============================================================================
print("\n" + "=" * 65)
print("  PHASE C: VISUALIZATION")
print("=" * 65)

# ── Plot 1: Hyperparameter Impact — learning_rate vs Accuracy (AdaBoost) ─────
print("\n  Generating Plot 1: Hyperparameter Impact...")
lr_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
train_scores, val_scores = [], []
for lr in lr_range:
    clf = AdaBoostClassifier(
        n_estimators=ada_grid.best_params_["n_estimators"],
        learning_rate=lr, random_state=7
    )
    clf.fit(X_train_sc, y_train)
    train_scores.append(accuracy_score(y_train, clf.predict(X_train_sc)))
    val_scores.append(accuracy_score(y_test,  clf.predict(X_test_sc)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lr_range, train_scores, "o-", color=COLORS[0], lw=2, label="Training Accuracy")
ax.plot(lr_range, val_scores,   "s--", color=COLORS[1], lw=2, label="Validation Accuracy")
ax.set_xlabel("Learning Rate", fontsize=12); ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Hyperparameter Impact: Learning Rate vs Accuracy\n(AdaBoost Classifier)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a2_plot1_hyperparam_impact.png", dpi=150)
plt.close()
print("  ✔ Plot 1 saved: kripa_a2_plot1_hyperparam_impact.png")

# ── Plot 2: Grouped Bar Chart ─────────────────────────────────────────────────
print("  Generating Plot 2: Model Comparison Matrix...")
metric_names = ["Accuracy","F1-Score","ROC-AUC"]
model_names  = list(results.keys())
x = np.arange(len(model_names)); width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, (metric, color) in enumerate(zip(metric_names, COLORS)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x+i*width, vals, width, label=metric, color=color, alpha=0.9)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Model", fontsize=12); ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Accuracy, F1-Score & ROC-AUC (Tuned Models)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x+width); ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0,1.10); ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a2_plot2_model_comparison.png", dpi=150)
plt.close()
print("  ✔ Plot 2 saved: kripa_a2_plot2_model_comparison.png")

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
print("  Generating Plot 3: Confusion Matrix Heatmap...")
cm = confusion_matrix(y_test, results[best_name]["y_pred"])
tn,fp,fn,tp = cm.ravel()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=raw.target_names, yticklabels=raw.target_names,
            linewidths=0.5, linecolor="gray", ax=ax, annot_kws={"size":16,"weight":"bold"})
ax.set_xlabel("Predicted Label", fontsize=12); ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — {best_name} (Best Ensemble)\nTP={tp}  TN={tn}  FP={fp}  FN={fn}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/kripa_a2_plot3_confusion_matrix.png", dpi=150)
plt.close()
print("  ✔ Plot 3 saved: kripa_a2_plot3_confusion_matrix.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
summary = pd.DataFrame({m:{k:f"{v:.4f}" for k,v in res.items() if k in metric_names} for m,res in results.items()}).T
summary.index.name = "Model"
print(f"\n{summary.to_string()}")
print(f"\n  Best Model (ROC-AUC): {best_name}")
print(f"  Accuracy   : {results[best_name]['Accuracy']:.4f}")
print(f"  F1-Score   : {results[best_name]['F1-Score']:.4f}")
print(f"  ROC-AUC    : {results[best_name]['ROC-AUC']:.4f}")
print("\n  ✔ Coding Assignment 2 (Kripa Das) — Complete!\n")
