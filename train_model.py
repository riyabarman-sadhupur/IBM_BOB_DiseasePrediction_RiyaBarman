"""
Disease Prediction - Model Training Script
=========================================
Dataset  : Synthetically generated cardiovascular risk dataset (1200 rows)
Models   : Logistic Regression, Random Forest (pure-numpy/pandas),
           plus scikit-learn Gradient Boosting if available
Artifacts: models/best_model.pkl, models/scaler.pkl,
           models/model_metadata.json, data/dataset.csv
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from models_def import (
    SimpleScaler, LogisticRegressionNP, RandomForestNP,
    DecisionTreeBinary, sigmoid
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "report_assets")

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

FEATURES = [
    "age", "sex", "bmi", "blood_pressure", "cholesterol",
    "glucose", "smoking", "family_history", "chest_pain_type", "heart_rate"
]

# ─── 1. Dataset generation ────────────────────────────────────────────────────
def generate_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age         = rng.integers(25, 80, n_samples).astype(float)
    sex         = rng.integers(0, 2, n_samples).astype(float)
    bmi         = np.clip(rng.normal(27.5, 5.0, n_samples), 15, 50).round(1)
    bp          = np.clip(rng.normal(120, 20, n_samples), 80, 200).round(1)
    cholesterol = np.clip(rng.normal(200, 40, n_samples), 120, 350).round(1)
    glucose     = np.clip(rng.normal(100, 25, n_samples), 60, 300).round(1)
    smoking     = rng.integers(0, 2, n_samples).astype(float)
    family_hist = rng.integers(0, 2, n_samples).astype(float)
    chest_pain  = rng.integers(0, 4, n_samples).astype(float)
    heart_rate  = np.clip(rng.normal(75, 12, n_samples), 45, 130).round(1)

    risk = (
        0.03 * (age - 40) + 0.5 * sex + 0.02 * (bmi - 25)
        + 0.008 * (bp - 100) + 0.005 * (cholesterol - 180)
        + 0.006 * (glucose - 90) + 0.4 * smoking
        + 0.3 * family_hist + 0.2 * chest_pain
        + rng.normal(0, 0.5, n_samples)
    )
    target = (risk > 2.2).astype(int)

    return pd.DataFrame({
        "age": age, "sex": sex, "bmi": bmi,
        "blood_pressure": bp, "cholesterol": cholesterol,
        "glucose": glucose, "smoking": smoking,
        "family_history": family_hist, "chest_pain_type": chest_pain,
        "heart_rate": heart_rate, "target": target
    })

# ─── 2. Metrics ───────────────────────────────────────────────────────────────
def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def f1(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))

def roc_auc(y_true, y_score):
    order = np.argsort(-y_score)
    yt = y_true[order]; ys = y_score[order]
    tp, fp = 0, 0
    n_pos = y_true.sum(); n_neg = len(y_true) - n_pos
    auc = 0.0
    prev_tp, prev_fp = 0, 0
    for i in range(len(yt)):
        if i > 0 and ys[i] != ys[i-1]:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2
            prev_tp, prev_fp = tp, fp
        if yt[i]: tp += 1
        else:     fp += 1
    auc += (fp - prev_fp) * (tp + prev_tp) / 2
    return float(auc / (n_pos * n_neg + 1e-8))

def confusion_matrix_np(y_true, y_pred):
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return [[tn, fp], [fn, tp]]

def cross_val_accuracy(model_class, X, y, k=5, **kwargs):
    n = len(X)
    fold = n // k
    scores = []
    for i in range(k):
        val_idx   = slice(i * fold, (i + 1) * fold)
        train_idx = list(range(0, i * fold)) + list(range((i + 1) * fold, n))
        m = model_class(**kwargs)
        m.fit(X[train_idx], y[train_idx])
        scores.append(accuracy(y[val_idx], m.predict(X[val_idx])))
    return float(np.mean(scores)), float(np.std(scores))

# ─── 6. Train & evaluate ──────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    print("  Training Logistic Regression ...")
    lr_model = LogisticRegressionNP(lr=0.1, n_iter=400, lam=1e-4)
    lr_model.fit(X_train, y_train)

    print("  Training Random Forest (50 trees) ...")
    rf_model = RandomForestNP(n_estimators=50, max_depth=6, seed=42)
    rf_model.fit(X_train, y_train)

    results = {}
    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba_1(X_test)
        acc  = accuracy(y_test, y_pred)
        f1s  = f1(y_test, y_pred)
        auc  = roc_auc(y_test, y_proba)
        results[name] = {
            "model": model,
            "accuracy":  round(acc, 4),
            "f1":        round(f1s, 4),
            "roc_auc":   round(auc, 4),
            "cv_mean":   0.0,
            "cv_std":    0.0,
            "conf_mat":  confusion_matrix_np(y_test, y_pred),
        }
        print(f"    {name}: acc={acc:.4f}  F1={f1s:.4f}  AUC={auc:.4f}")

    # Cross-val scores for LR
    cv_mean, cv_std = cross_val_accuracy(
        LogisticRegressionNP, X_train, y_train, k=5,
        lr=0.1, n_iter=400, lam=1e-4
    )
    results["Logistic Regression"]["cv_mean"] = round(cv_mean, 4)
    results["Logistic Regression"]["cv_std"]  = round(cv_std,  4)

    return results

# ─── 7. Plots ─────────────────────────────────────────────────────────────────
def eda_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["target"].value_counts()
    axes[0].bar(["No Disease", "Disease"],
                [counts.get(0, 0), counts.get(1, 0)],
                color=["#3b82d4", "#e05c5c"])
    axes[0].set_title("Target Distribution")
    axes[0].set_ylabel("Count")

    df[df.target == 0]["age"].plot.hist(ax=axes[1], alpha=0.6,
                                        color="#3b82d4", label="No Disease", bins=20)
    df[df.target == 1]["age"].plot.hist(ax=axes[1], alpha=0.6,
                                        color="#e05c5c", label="Disease", bins=20)
    axes[1].set_title("Age Distribution by Target")
    axes[1].set_xlabel("Age")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "eda_distribution.png"), dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="Blues", ax=ax, linewidths=0.4, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()
    print("  [OK] EDA plots saved")

def plot_model_comparison(results):
    names = list(results.keys())
    acc   = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1"]       for n in names]
    auc   = [results[n]["roc_auc"]  for n in names]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, acc, w, label="Accuracy", color="#3b82d4")
    ax.bar(x,     f1s, w, label="F1 Score",  color="#7c5cd8")
    ax.bar(x + w, auc, w, label="ROC-AUC",   color="#22c55e")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Score")
    ax.set_title("Model Comparison"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    print("  [OK] Model comparison plot saved")

def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()
    print("  [OK] Confusion matrix plot saved")

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx], imp[idx], color="#3b82d4")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=120)
    plt.close()
    print("  [OK] Feature importance plot saved")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Disease Prediction — Model Training")
    print("=" * 60)

    print("\n[1] Generating dataset ...")
    df = generate_dataset()
    df.to_csv(os.path.join(DATA_DIR, "dataset.csv"), index=False)
    print(f"  [OK] Dataset: {df.shape[0]} rows × {df.shape[1]} cols  "
          f"| Disease prevalence: {df.target.mean()*100:.1f}%")

    print("\n[2] Running EDA ...")
    eda_plots(df)

    print("\n[3] Preparing features ...")
    X = df[FEATURES].values.astype(float)
    y = df["target"].values.astype(int)

    n = len(X)
    test_n = int(n * 0.2)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    test_idx, train_idx = idx[:test_n], idx[test_n:]

    scaler = SimpleScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s  = scaler.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  [OK] Train: {len(train_idx)}  |  Test: {len(test_idx)}")

    print("\n[4] Training models ...")
    results = train_models(X_train_s, X_test_s, y_train, y_test)

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best = results[best_name]
    print(f"\n  * Best model: {best_name}  (AUC={best['roc_auc']:.4f})")

    print("\n[5] Generating plots ...")
    plot_model_comparison(results)
    plot_confusion_matrix(best["conf_mat"], best_name)
    plot_feature_importance(best["model"], FEATURES, best_name)

    print("\n[6] Saving artifacts ...")
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("  [OK] best_model.pkl, scaler.pkl saved")

    metadata = {
        "best_model_name": best_name,
        "features":        FEATURES,
        "accuracy":        best["accuracy"],
        "f1_score":        best["f1"],
        "roc_auc":         best["roc_auc"],
        "cv_mean":         best["cv_mean"],
        "cv_std":          best["cv_std"],
        "confusion_matrix": best["conf_mat"],
        "all_models": {
            n: {"accuracy": r["accuracy"], "f1": r["f1"], "roc_auc": r["roc_auc"]}
            for n, r in results.items()
        },
        "dataset_info": {
            "total_samples": int(df.shape[0]),
            "n_features":    int(len(FEATURES)),
            "disease_count": int(df.target.sum()),
            "healthy_count": int((df.target == 0).sum()),
        }
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("  [OK] model_metadata.json saved")

    print("\n" + "=" * 60)
    print(" Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
