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
    print(f"  Class balance — correct: {y.sum():,}  wrong: {(y==0).sum():,}")
    return X, y
 
 