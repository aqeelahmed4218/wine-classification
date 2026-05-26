# 🍷 Wine Quality Dataset — Multi-Model Classification

> **Student:** Aqeel Ahmed
> **Roll No:** F22BSEEN1E02085
> **Course Assignment — Machine Learning**

---

## 📌 Project Overview

This project implements and compares **four classification models** on the Red Wine Quality dataset. It covers the full ML pipeline: data collection → preprocessing → model training → evaluation → visualisation.

The wine quality scores (3–8) are converted into a **binary classification** task:
- **Class 0** — Low Quality (score 3–5)
- **Class 1** — High Quality (score 6–9)

---

## 📂 Project Structure

```
wine_classification/
│
├── wine_classification.py    # Main script (full pipeline)
├── winequality-red.csv       # Dataset (~150 KB, 1599 samples)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── plots/
    ├── plot1_eda.png                # Exploratory Data Analysis
    ├── plot2_model_comparison.png   # Accuracy & CV comparison
    ├── plot3_confusion_matrices.png # Confusion matrices (all models)
    ├── plot4_feature_importance.png # Random Forest feature importance
    └── plot5_report_heatmap.png     # Per-class precision/recall/F1
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Name | Red Wine Quality |
| Source | UCI Machine Learning Repository |
| Samples | 1,599 |
| Features | 11 physicochemical properties |
| Target | Quality score (3–8) → binary class |
| File Size | ~150 KB (production sensor datasets scale to GBs) |

### Features
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free & Total sulfur dioxide
- Density, pH, Sulphates, **Alcohol**

---

## 🤖 Models Implemented

| Model | Test Accuracy | CV Score (5-fold) |
|---|---|---|
| **Random Forest** | **79.0%** ⭐ | **80.8%** |
| SVM (RBF kernel) | 75.5% | 76.4% |
| Logistic Regression | 73.5% | 74.4% |
| Decision Tree | 73.8% | 73.0% |

**Best model: Random Forest with 79.0% test accuracy**

> Key finding: **Alcohol content** and **volatile acidity** are the most important predictors of wine quality.

---

## 🔧 Steps Followed

1. **Collect Dataset** — Downloaded from UCI repository (CSV, ~150 KB; scales to GBs in production)
2. **Preprocessing** — Binary label encoding, StandardScaler normalisation, 75/25 stratified split
3. **Read / Explore** — Shape, statistics, class distribution, scatter & correlation plots
4. **Classification Models** — Random Forest, SVM, Logistic Regression, Decision Tree
5. **Results & Graphs** — 5 publication-quality plots saved as PNG files

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/wine-classification.git
cd wine-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the script
python wine_classification.py
```

All 5 plots will be saved in the working directory.

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 📜 License

Open Source — MIT License. Free to use and modify.

---

*Aqeel Ahmed | F22BSEEN1E02085*
