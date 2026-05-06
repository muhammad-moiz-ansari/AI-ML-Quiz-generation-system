"""
model_a_train.py
================
Model A - Answer Verifier + Question Generator
"""

from scipy.stats import false_discovery_control
import os
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.preprocessing import LabelEncoder
 
# ── Paths ───────────────────────────────────────────────────────────────────
is_local = False

BASE_DIR = r"/content/drive/MyDrive/Colab Notebooks/AI PROJECT/"

if is_local:
    PROCESSED_DIR   = "../data/processed/"
    MODELS_DIR      = "../models/model_a/traditional/"
    PLOTS_DIR       = "../notebooks/plots/"
else:
    PROCESSED_DIR   = os.path.join(BASE_DIR, "data/processed/")
    MODELS_DIR      = os.path.join(BASE_DIR, "models/model_a/traditional/")
    PLOTS_DIR       = os.path.join(BASE_DIR, "notebooks/plots/")
 
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
def load_data(split="train"):
    """Load cleaned CSV and the shared One-Hot vectorizer."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"clean_{split}.csv"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/onehot_encoder.pkl"))
    return df, vectorizer
 
 
def expand_to_verification_pairs(df, vectorizer, max_rows=None):
    """
    Convert each RACE row into 4 binary classification examples.
    """
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)
 
    texts  = []   # combined text strings
    labels = []   # 0 or 1
 
    option_cols = ['A', 'B', 'C', 'D']
 
    for _, row in df.iterrows():
        for opt in option_cols:
            # Build the combined text the model sees
            combined = (
                str(row.get('clean_article',  '')) + ' ' +
                str(row.get('clean_question', '')) + ' ' +
                str(row.get(f'clean_{opt}',  ''))
            )
            texts.append(combined)
 
            # Label: 1 if this option is the correct answer, else 0
            label = 1 if row['answer'] == opt else 0
            labels.append(label)
 
    X = vectorizer.transform(texts)          # sparse OHE matrix
    y = np.array(labels, dtype=int)
 
    print(f"  Expanded to {X.shape[0]:,} examples  |  features: {X.shape[1]:,}")
    print(f"  Class balance - correct: {y.sum():,}  wrong: {(y==0).sum():,}")
    return X, y
 


# ── Evaluation Helpers ───────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name):
    """Save a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Wrong', 'Predicted Correct'])
    ax.set_yticklabels(['Actually Wrong',  'Actually Correct'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved → {path}")
 
 
def evaluate(model, X_dev, y_dev, model_name):
    """Print and return evaluation metrics."""
    y_pred = model.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    f1_macro = f1_score(y_dev, y_pred, average='macro')
    cm = confusion_matrix(y_dev, y_pred)
 
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1  : {f1_macro:.4f}")
    print(classification_report(y_dev, y_pred,
                                target_names=['Wrong', 'Correct']))
    plot_confusion_matrix(cm, model_name)
 
    return {"model": model_name, "accuracy": acc, "macro_f1": f1_macro}
 
 
# ── SECTION 1: Supervised Models ─────────────────────────────────────────────
 
def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression
    -------------------
    Learns a weight for each of the 5000 OHE features.
    C = regularisation strength (smaller → more regularised → simpler model).
    """
    print("\n[1/4] Training Logistic Regression...")
    t0  = time.time()
    model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model
 
 
def train_svm(X_train, y_train):
    """
    Support Vector Machine (LinearSVC)
    -----------------------------------
    We wrap it in CalibratedClassifierCV so it can output probabilities
    C = regularisation
    """
    print("\n[2/4] Training SVM (LinearSVC + Calibration)...")
    t0  = time.time()
    base  = LinearSVC(C=0.1, max_iter=2000, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)   # adds predict_proba
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model
 
 
def train_naive_bayes(X_train, y_train):
    """
    Bernoulli Naive Bayes
    ----------------------
    BernoulliNB is designed for binary/boolean features
    alpha = Laplace smoothing
    """
    print("\n[3/4] Training Bernoulli Naive Bayes...")
    t0  = time.time()
    model = BernoulliNB(alpha=1.0)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model
 
 
 