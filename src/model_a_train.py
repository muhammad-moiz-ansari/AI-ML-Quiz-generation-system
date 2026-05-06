"""
model_a_train.py
================
Model A - Answer Verifier + Question Generator
"""

from scipy.stats import false_discovery_control
import os
import time
import string
import re
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
DO_FULL_TRAIN = False		# Set to True for full training

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
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42, n_jobs=-1)
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
    base  = LinearSVC(C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
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
 
 
 
# ── SECTION 2: Unsupervised - K-Means ────────────────────────────
 
def train_kmeans(X_train, y_train):
    """
    K-Means Clustering (Unsupervised)
    ----------------------------------
    K-Means does NOT use labels during training.
    """
    print("\n[4/4] Training K-Means Clustering (Unsupervised)...")
 
    # Sparse matrices need special handling for KMeans
    # Convert a subsample to dense for speed (KMeans is slow on sparse rows)
    print("  Subsampling 20,000 examples for K-Means (memory efficient)...")
 
    # Random subsample
    rng = np.random.RandomState(42)
    indices = rng.choice(X_train.shape[0], size=min(20000, X_train.shape[0]), replace=False)
    X_sub = X_train[indices].toarray()   # convert sparse → dense for KMeans
    y_sub = y_train[indices]
 
    t0 = time.time()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_sub)
    print(f"  Trained in {time.time()-t0:.1f}s")
 
    # -- Evaluate clustering --------------------------------------------------
    labels_pred = kmeans.labels_
 
    # Silhouette score (expensive on large arrays - use subsample of 3000)
    sil_idx = rng.choice(len(X_sub), size=min(3000, len(X_sub)), replace=False)
    sil_score = silhouette_score(X_sub[sil_idx], labels_pred[sil_idx])
    print(f"\n  Silhouette Score : {sil_score:.4f}  (range: -1 to 1, higher is better)")
 
    # Cluster purity
    for cluster_id in [0, 1]:
        mask = labels_pred == cluster_id
        cluster_y = y_sub[mask]
        if len(cluster_y) == 0:
            continue
        majority = cluster_y.mean()   # fraction that are "correct" (label=1)
        purity = max(majority, 1 - majority)
        print(f"  Cluster {cluster_id}: {mask.sum():,} samples  |  "
              f"purity={purity:.3f}  |  "
              f"{'mostly CORRECT answers' if majority > 0.5 else 'mostly WRONG answers'}")
 
    return kmeans, sil_score
 
 
# ── SECTION 3: Semi-Supervised - Label Propagation ──────────────────────────
 
def train_label_propagation(X_train, y_train):
    """
    Label Propagation (Semi-Supervised)
    -------------------------------------
    Uses a SMALL labeled set + a LARGE unlabeled set.
    Unlabeled examples are marked with label = -1.
    """
    print("\n[5/4] Training Label Propagation (Semi-Supervised)...")
 
    rng     = np.random.RandomState(42)
    indices = rng.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
    X_sub   = X_train[indices].toarray()
    y_sub   = y_train[indices].copy()
 
    # Mask 80% of labels → pretend they're unlabeled
    mask_unlabeled         = rng.rand(len(y_sub)) < 0.80
    y_semi                 = y_sub.copy()
    y_semi[mask_unlabeled] = -1     # -1 = unlabeled in sklearn
 
    t0 = time.time()
    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp.fit(X_sub, y_semi)
    print(f"  Trained in {time.time()-t0:.1f}s")
 
    # Evaluate only on the samples that were originally labeled
    labeled_mask = ~mask_unlabeled
    y_true_lp = y_sub[labeled_mask]
    y_pred_lp = lp.predict(X_sub[labeled_mask])
 
    acc_lp = accuracy_score(y_true_lp, y_pred_lp)
    f1_lp = f1_score(y_true_lp, y_pred_lp, average='macro')
    print(f"  Label Propagation Accuracy : {acc_lp:.4f}")
    print(f"  Label Propagation Macro F1 : {f1_lp:.4f}")
 
    return lp
 
 
# ── SECTION 4: Ensemble ──────────────────────────────────────────────────────
 
def train_ensemble(lr_model, svm_model, nb_model, X_train, y_train):
    """
    Soft-Vote Ensemble
    -------------------
    Averages the predict_proba() output of all 3 classifiers.
    """
    print("\n[ENSEMBLE] Building Soft-Vote Ensemble (LR + SVM + NB)...")
 
    ensemble = VotingClassifier(
        estimators=[
            ('lr',  lr_model),
            ('svm', svm_model),
            ('nb',  nb_model),
        ],
        voting='soft',      # use predicted probabilities
        weights=[1, 1, 1],  # equal weight - can tune later
    )
 
    # VotingClassifier needs to re-fit. We pass X_train again.
    # This is fast because it just wraps already-fitted estimators.
    t0 = time.time()
    ensemble.fit(X_train, y_train)
    print(f"  Ensemble built in {time.time()-t0:.1f}s")
    return ensemble
 
 

# ── SECTION 5: Question Generator (Template-Based) ──────────────────────────
 
def generate_question(article_text, correct_answer_text):
    """
    Template-Based Question Generation
    ------------------------------------
    This is a rule-based approach - no ML needed for the generator itself.
    The ML ranker (SVM) scores question quality.
    """
 
    STOPWORDS = {'the','a','an','is','was','are','were','to','of',
                 'in','for','on','with','at','by','from','it','its'}
 
    def clean(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation))
 
    # Get key words from the correct answer
    answer_words = set(clean(correct_answer_text).split()) - STOPWORDS
 
    # Split article into sentences
    sentences = re.split(r'(?<=[.!?])\s+', article_text)
 
    best_sentence = None
    best_overlap  = 0
 
    for sent in sentences:
        sent_words = set(clean(sent).split()) - STOPWORDS
        overlap    = len(answer_words & sent_words)
        if overlap > best_overlap:
            best_overlap  = overlap
            best_sentence = sent
 
    if not best_sentence:
        return "What is the main idea of the passage?"
 
    # Apply simple Wh-word template
    # Replace the answer phrase in the sentence with a blank, then prepend "What"
    pattern  = re.compile(re.escape(correct_answer_text), re.IGNORECASE)
    question = pattern.sub("_____", best_sentence)
    question = "What " + question.strip().lstrip('The the A a').strip()
 
    # Trim to reasonable length
    if len(question.split()) > 20:
        question = ' '.join(question.split()[:18]) + "?"
 
    return question
 
 
# ══ MAIN ═════════════════════════════════════════════════════════════════════
 
def main():
    print("=" * 60)
    print("  MODEL A - Answer Verifier Training")
    print("=" * 60)
 
    # ── Load data ──────────────────────────────────────────────────────────
    print("\n>>> Loading training data...")
    train_df, vectorizer = load_data("train")
    dev_df,   _          = load_data("dev")    # vectorizer already loaded
 
    # ── Build feature matrices ─────────────────────────────────────────────
    # Determine row limits based on DO_FULL_TRAIN flag
    train_limit = None if DO_FULL_TRAIN else 20000
    dev_limit   = None if DO_FULL_TRAIN else 5000
    str_temp    = "Full dataset" if DO_FULL_TRAIN else "Small dataset"

    print(f"\n>>> Expanding training set to verification pairs ({str_temp})...")
    X_train, y_train = expand_to_verification_pairs(train_df, vectorizer, max_rows=train_limit)
 
    print("\n>>> Expanding dev set...")
    X_dev,   y_dev   = expand_to_verification_pairs(dev_df,   vectorizer, max_rows=dev_limit)
 
    # ── Train supervised models ────────────────────────────────────────────
    lr_model  = train_logistic_regression(X_train, y_train)
    svm_model = train_svm(X_train, y_train)
    nb_model  = train_naive_bayes(X_train, y_train)
 
    # ── Evaluate supervised models ─────────────────────────────────────────
    print("\n>>> Evaluating on dev set...")
    results = []
    results.append(evaluate(lr_model,  X_dev, y_dev, "Logistic Regression"))
    results.append(evaluate(svm_model, X_dev, y_dev, "SVM (LinearSVC)"))
    results.append(evaluate(nb_model,  X_dev, y_dev, "Bernoulli Naive Bayes"))
 
    # ── Unsupervised: K-Means ──────────────────────────────────────────────
    print("\n>>> Running K-Means Clustering (Unsupervised)...")
    kmeans_model, sil_score = train_kmeans(X_train, y_train)
 
    # ── Semi-Supervised: Label Propagation ────────────────────────────────
    lp_model = train_label_propagation(X_train, y_train)
 
    # ── Ensemble ───────────────────────────────────────────────────────────
    ensemble_model = train_ensemble(lr_model, svm_model, nb_model, X_train, y_train)
    results.append(evaluate(ensemble_model, X_dev, y_dev, "Soft-Vote Ensemble"))
 
    # ── Comparison table ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FINAL COMPARISON TABLE")
    print("═" * 60)
    results_df = pd.DataFrame(results)
    results_df['accuracy'] = results_df['accuracy'].map('{:.4f}'.format)
    results_df['macro_f1'] = results_df['macro_f1'].map('{:.4f}'.format)
    print(results_df.to_string(index=False))
    print(f"\n  K-Means Silhouette Score : {sil_score:.4f}")
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_a_results.csv'), index=False)
 
    # ── Save all models ────────────────────────────────────────────────────
    print("\n>>> Saving models to", MODELS_DIR)
    joblib.dump(lr_model,       os.path.join(MODELS_DIR, 'lr_model.pkl'))
    joblib.dump(svm_model,      os.path.join(MODELS_DIR, 'svm_model.pkl'))
    joblib.dump(nb_model,       os.path.join(MODELS_DIR, 'nb_model.pkl'))
    joblib.dump(kmeans_model,   os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    joblib.dump(lp_model,       os.path.join(MODELS_DIR, 'lp_model.pkl'))
    joblib.dump(ensemble_model, os.path.join(MODELS_DIR, 'ensemble_model.pkl'))
    print("All models saved successfully!")
 
    # ── Bar chart comparison ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    models  = [r['model'] for r in results]
    accs    = [float(r['accuracy']) for r in results]
    f1s     = [float(r['macro_f1']) for r in results]
    x       = np.arange(len(models))
    width   = 0.35
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Model A - Supervised Model Comparison', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_a_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Comparison chart saved!")
 
    print("\n✅ Model A training complete!")
 
 
if __name__ == "__main__":
    main()