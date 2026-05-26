# ============================================================
#  Wine Quality Dataset - Multi-Model Classification
#  Student : Aqeel Ahmed
#  Roll No : F22BSEEN1E02085
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ── 2. LOAD & INSPECT ────────────────────────────────────────
print("=" * 62)
print("  WINE QUALITY CLASSIFICATION — Aqeel Ahmed (F22BSEEN1E02085)")
print("=" * 62)

df = pd.read_csv("winequality-red.csv")
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    File size        ≈  {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"    (Production wine datasets from IoT sensors scale to GBs)")
print(f"\n[2] First 5 rows:\n{df.head()}")
print(f"\n[3] Basic statistics:\n{df.describe().round(2)}")
print(f"\n[4] Quality score distribution:\n{df['quality'].value_counts().sort_index()}")

# ── 3. PREPROCESSING ─────────────────────────────────────────
print("\n[5] Preprocessing …")

# Binarise quality: low (3-5) = 0, high (6-9) = 1
df["quality_class"] = (df["quality"] >= 6).astype(int)
class_names = ["Low Quality (3-5)", "High Quality (6-9)"]

X = df.drop(columns=["quality", "quality_class"])
y = df["quality_class"]

print(f"    Class 0 (Low ) : {(y==0).sum()} samples")
print(f"    Class 1 (High) : {(y==1).sum()} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"    Train : {X_train_sc.shape}  |  Test : {X_test_sc.shape}")

# ── 4. TRAIN FOUR MODELS ─────────────────────────────────────
print("\n[6] Training models …")
models = {
    "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM":                 SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
}

palette = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D"]
results = {}
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(clf, X_train_sc, y_train, cv=5).mean()
    results[name] = {"model": clf, "y_pred": y_pred,
                     "accuracy": acc, "cv_score": cv}
    print(f"    {name:<22}  acc={acc:.3f}  cv={cv:.3f}")

best_name = max(results, key=lambda k: results[k]["accuracy"])
print(f"\n  ★  Best model: {best_name} "
      f"({results[best_name]['accuracy']*100:.1f}% accuracy)")

# ── 5. VISUALISATIONS ────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Fig 1: EDA ───────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle("Wine Quality Dataset — Exploratory Data Analysis",
              fontsize=14, fontweight="bold")

# Quality score distribution
quality_counts = df["quality"].value_counts().sort_index()
axes[0].bar(quality_counts.index, quality_counts.values,
            color=palette[0], alpha=0.85, edgecolor="white", linewidth=0.8)
axes[0].set_xlabel("Quality Score"); axes[0].set_ylabel("Count")
axes[0].set_title("Quality Score Distribution")
for x, v in zip(quality_counts.index, quality_counts.values):
    axes[0].text(x, v + 5, str(v), ha="center", fontsize=9)

# Alcohol vs Quality
colors = [palette[0] if c == 0 else palette[1] for c in df["quality_class"]]
axes[1].scatter(df["alcohol"], df["volatile acidity"], c=colors, alpha=0.5, s=25)
axes[1].set_xlabel("Alcohol (%)"); axes[1].set_ylabel("Volatile Acidity")
axes[1].set_title("Alcohol vs Volatile Acidity")
p0 = mpatches.Patch(color=palette[0], label="Low Quality")
p1 = mpatches.Patch(color=palette[1], label="High Quality")
axes[1].legend(handles=[p0, p1])

# Correlation heatmap (top features)
corr = X.corrwith(y).abs().sort_values(ascending=False)
axes[2].barh(corr.index, corr.values, color=palette[3], alpha=0.85)
axes[2].set_xlabel("Absolute Correlation with Quality")
axes[2].set_title("Feature Correlation with Quality")
axes[2].invert_yaxis()

plt.tight_layout()
fig1.savefig("plot1_eda.png", dpi=150, bbox_inches="tight")
print("\n  Saved: plot1_eda.png")

# ── Fig 2: Model Accuracy Comparison ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
cvs   = [results[n]["cv_score"]  * 100 for n in names]
x = np.arange(len(names)); w = 0.35
bars1 = ax2.bar(x - w/2, accs, w, label="Test Accuracy",  color=palette, alpha=0.9)
bars2 = ax2.bar(x + w/2, cvs,  w, label="CV Score (5-fold)", color=palette,
                alpha=0.45, hatch="//")
ax2.set_ylim(60, 105)
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=12)
ax2.set_ylabel("Accuracy (%)");
ax2.set_title("Model Comparison — Test Accuracy vs CV Score", fontweight="bold")
ax2.legend()
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig2.savefig("plot2_model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: plot2_model_comparison.png")

# ── Fig 3: Confusion Matrices ─────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
for ax, (name, res) in zip(axes3.flat, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Low", "High"])
    disp.plot(ax=ax, colorbar=False, cmap="Reds")
    ax.set_title(f"{name}  (acc={res['accuracy']*100:.1f}%)", fontweight="bold")
plt.tight_layout()
fig3.savefig("plot3_confusion_matrices.png", dpi=150, bbox_inches="tight")
print("  Saved: plot3_confusion_matrices.png")

# ── Fig 4: Feature Importance (Random Forest) ────────────────
fig4, ax4 = plt.subplots(figsize=(9, 5))
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_,
                        index=X.columns).sort_values()
colors_imp = [palette[0] if v < importances.median() else palette[1]
              for v in importances.values]
importances.plot.barh(ax=ax4, color=colors_imp, alpha=0.88)
ax4.set_title("Feature Importances — Random Forest", fontweight="bold")
ax4.set_xlabel("Importance Score")
ax4.axvline(importances.median(), color="gray", linestyle="--",
            linewidth=1, label="Median")
ax4.legend()
plt.tight_layout()
fig4.savefig("plot4_feature_importance.png", dpi=150, bbox_inches="tight")
print("  Saved: plot4_feature_importance.png")

# ── Fig 5: Precision / Recall / F1 Heatmap ───────────────────
fig5, axes5 = plt.subplots(1, 4, figsize=(18, 4))
fig5.suptitle("Per-Class F1 / Precision / Recall Heatmap",
              fontsize=13, fontweight="bold")
for ax, (name, res) in zip(axes5, results.items()):
    rpt = classification_report(y_test, res["y_pred"],
                                target_names=["Low", "High"],
                                output_dict=True)
    df_rpt = pd.DataFrame(rpt).T.iloc[:2][["precision", "recall", "f1-score"]]
    sns.heatmap(df_rpt.astype(float), annot=True, fmt=".2f",
                cmap="YlOrRd", vmin=0.6, vmax=1.0,
                ax=ax, linewidths=0.5)
    ax.set_title(name, fontsize=9)
plt.tight_layout()
fig5.savefig("plot5_report_heatmap.png", dpi=150, bbox_inches="tight")
print("  Saved: plot5_report_heatmap.png")

# ── Final Report ──────────────────────────────────────────────
print(f"\n[7] Classification Report — Best Model ({best_name}):")
print(classification_report(y_test, results[best_name]["y_pred"],
                            target_names=["Low Quality", "High Quality"]))
print("=" * 62)
print("  DONE — Aqeel Ahmed | F22BSEEN1E02085")
print("=" * 62)
