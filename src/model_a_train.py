"""
model_a_train.py
================
Model A - Answer Verifier + Question Generator
"""

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
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.utils.class_weight import compute_sample_weight

# ── NLP evaluation metrics (pip install nltk rouge-score) ───────────────────
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer
    # Download required NLTK data quietly
    nltk.download('wordnet',       quiet=True)
    nltk.download('punkt',         quiet=True)
    nltk.download('punkt_tab',     quiet=True)
    nltk.download('omw-1.4',       quiet=True)
    NLP_METRICS_AVAILABLE = True
    print("[INFO] NLTK + rouge-score loaded - NLP metrics enabled.")
except ImportError:
    NLP_METRICS_AVAILABLE = False
    print("[WARNING] nltk or rouge-score not installed.")
    print("         Run: pip install nltk rouge-score")
    print("         NLP generation metrics will be skipped.")

# ── Paths ────────────────────────────────────────────────────────────────────
is_local = False
DO_FULL_TRAIN = True

BASE_DIR = r"/content/drive/MyDrive/Colab Notebooks/AI PROJECT/"

if is_local:
    PROCESSED_DIR = "../data/processed/"
    MODELS_DIR    = "../models/model_a/traditional/"
    PLOTS_DIR     = "../notebooks/plots/"
else:
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/")
    MODELS_DIR    = os.path.join(BASE_DIR, "models/model_a/traditional/")
    PLOTS_DIR     = os.path.join(BASE_DIR, "notebooks/plots/")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

STOPWORDS = {
    'the','a','an','is','was','are','were','to','of','in','for','on',
    'with','at','by','from','it','its','this','that','and','or','but',
    'not','be','been','has','have','had','he','she','they','we','i',
    'you','do','did','will','would','could','should','as','if','than',
    'then','so','no','up','out','about','into'
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(split="train"):
    """Load cleaned CSV and the shared One-Hot vectorizer."""
    df         = pd.read_csv(os.path.join(PROCESSED_DIR, f"clean_{split}.csv"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/onehot_encoder.pkl"))
    return df, vectorizer


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Jaccard overlap of non-stopword tokens between two texts."""
    words_a = set(text_a.lower().split()) - STOPWORDS
    words_b = set(text_b.lower().split()) - STOPWORDS
    if not words_a and not words_b:
        return 0.0
    return len(words_a & words_b) / (len(words_a | words_b) + 1e-9)


def _hand_crafted_features(article: str, question: str, option: str,
                            all_options: list) -> np.ndarray:
    """
    Build a small vector of hand-crafted features that capture the RELATIONSHIP between the 
    article, question, and a candidate option. These features are concatenated onto the OHE 
    vector to give classifiers a richer signal beyond simple word presence.

    Features (7 dimensions):
      0  keyword overlap between option and article
      1  keyword overlap between option and question
      2  keyword overlap between question and article
      3  length of option (normalised by max option length in the group)
      4  option position in [A,B,C,D] list  (0.0, 0.33, 0.67, 1.0)
      5  whether option appears verbatim as a substring of the article
      6  ratio of option unique words that also appear in the article
    """
    art_clean  = article.lower().translate(str.maketrans('', '', string.punctuation))
    q_clean    = question.lower().translate(str.maketrans('', '', string.punctuation))
    opt_clean  = option.lower().translate(str.maketrans('', '', string.punctuation))

    # Feature 0: option-article keyword overlap
    f0 = _keyword_overlap(opt_clean, art_clean)

    # Feature 1: option-question keyword overlap
    f1 = _keyword_overlap(opt_clean, q_clean)

    # Feature 2: question-article keyword overlap
    f2 = _keyword_overlap(q_clean, art_clean)

    # Feature 3: normalised option length
    max_len = max(len(o.split()) for o in all_options) if all_options else 1
    f3 = len(option.split()) / (max_len + 1e-9)

    # Feature 4: position of this option among the 4 options (0 to 1)
    try:
        pos = all_options.index(option)
        f4  = pos / max(len(all_options) - 1, 1)
    except ValueError:
        f4 = 0.0

    # Feature 5: binary - is the exact option text found in the article?
    f5 = 1.0 if opt_clean in art_clean else 0.0

    # Feature 6: fraction of option words present in article
    opt_words = set(opt_clean.split()) - STOPWORDS
    art_words = set(art_clean.split()) - STOPWORDS
    f6 = len(opt_words & art_words) / (len(opt_words) + 1e-9) if opt_words else 0.0

    return np.array([f0, f1, f2, f3, f4, f5, f6], dtype=np.float32)


def expand_to_verification_pairs(df, vectorizer, max_rows=None):
    """
    Convert each RACE row into 4 binary classification examples.
    Each example combines:
      - OHE sparse vector of (article + question + option)
      - 7 hand-crafted relationship features (dense)
    Returns a horizontally-stacked sparse matrix.
    """
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=42)

    texts          = []
    labels         = []
    hc_feature_rows = []

    option_cols = ['A', 'B', 'C', 'D']

    for _, row in df.iterrows():
        article  = str(row.get('clean_article',  ''))
        question = str(row.get('clean_question', ''))
        all_opts = [str(row.get(f'clean_{o}', '')) for o in option_cols]

        for opt in option_cols:
            option_text = str(row.get(f'clean_{opt}', ''))

            combined = article + ' ' + question + ' ' + option_text
            texts.append(combined)

            label = 1 if row['answer'] == opt else 0
            labels.append(label)

            hc = _hand_crafted_features(article, question, option_text, all_opts)
            hc_feature_rows.append(hc)

    # OHE sparse matrix
    X_ohe = vectorizer.transform(texts)

    # Hand-crafted dense → sparse for horizontal stacking
    X_hc  = scipy.sparse.csr_matrix(np.array(hc_feature_rows, dtype=np.float32))

    # Concatenate OHE + hand-crafted features horizontally
    X     = scipy.sparse.hstack([X_ohe, X_hc], format='csr')
    y     = np.array(labels, dtype=int)

    print(f"  Expanded to {X.shape[0]:,} examples  |  features: {X.shape[1]:,}")
    print(f"  (OHE: {X_ohe.shape[1]:,}  +  hand-crafted: {X_hc.shape[1]})")
    print(f"  Class balance - correct: {y.sum():,}  wrong: {(y==0).sum():,}")
    return X, y


# ── Evaluation ───────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name):
    """Save a confusion matrix heatmap."""
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
    print(f"  Confusion matrix saved -> {path}")


def evaluate(model, X_dev, y_dev, model_name):
    """Print classification metrics and return as dict."""
    y_pred     = model.predict(X_dev)
    acc        = accuracy_score(y_dev, y_pred)
    f1_macro   = f1_score(y_dev, y_pred, average='macro')
    f1_correct = f1_score(y_dev, y_pred, pos_label=1, average='binary')
    cm         = confusion_matrix(y_dev, y_pred)

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy      : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1      : {f1_macro:.4f}")
    print(f"  F1 (Correct)  : {f1_correct:.4f}  <-- most important metric")
    print(classification_report(y_dev, y_pred, target_names=['Wrong', 'Correct']))
    plot_confusion_matrix(cm, model_name)

    return {
        "model"      : model_name,
        "accuracy"   : round(acc, 4),
        "macro_f1"   : round(f1_macro, 4),
        "f1_correct" : round(f1_correct, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1: SUPERVISED MODELS
# ════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression - unchanged, already uses class_weight='balanced'.
    C=1.0 gives a good bias/variance tradeoff on OHE + hand-crafted features.
    """
    print("\n[1/5] Training Logistic Regression...")
    t0    = time.time()
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model


def train_svm(X_train, y_train):
    """
    Support Vector Machine (LinearSVC + Calibration)
    -------------------------------------------------
    """
    print("\n[2/5] Training SVM (LinearSVC + Calibration)...")
    print("  Note: C raised from 0.1 to 1.0 to fix majority-class collapse.")
    t0    = time.time()
    base  = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model


def train_naive_bayes(X_train, y_train):
    """
    Bernoulli Naive Bayes
    ----------------------
    """
    print("\n[3/5] Training Bernoulli Naive Bayes (with balanced sample weights)...")
    t0     = time.time()
    model  = BernoulliNB(alpha=1.0)
    sw     = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, sample_weight=sw)
    print(f"  Trained in {time.time()-t0:.1f}s")
    return model


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2: UNSUPERVISED - K-Means  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def train_kmeans(X_train, y_train):
    """
    K-Means Clustering (Unsupervised)
    ----------------------------------
    """
    print("\n[4/5] Training K-Means Clustering (Unsupervised)...")
    print("  Subsampling 20,000 examples for K-Means (memory efficient)...")

    rng     = np.random.RandomState(42)
    indices = rng.choice(X_train.shape[0], size=min(20000, X_train.shape[0]), replace=False)
    X_sub   = X_train[indices].toarray()
    y_sub   = y_train[indices]

    t0     = time.time()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_sub)
    print(f"  Trained in {time.time()-t0:.1f}s")

    labels_pred = kmeans.labels_
    sil_idx     = rng.choice(len(X_sub), size=min(3000, len(X_sub)), replace=False)
    sil_score   = silhouette_score(X_sub[sil_idx], labels_pred[sil_idx])
    print(f"\n  Silhouette Score : {sil_score:.4f}  (range: -1 to 1, higher is better)")

    for cluster_id in [0, 1]:
        mask      = labels_pred == cluster_id
        cluster_y = y_sub[mask]
        if len(cluster_y) == 0:
            continue
        majority  = cluster_y.mean()
        purity    = max(majority, 1 - majority)
        print(f"  Cluster {cluster_id}: {mask.sum():,} samples  |  "
              f"purity={purity:.3f}  |  "
              f"{'mostly CORRECT answers' if majority > 0.5 else 'mostly WRONG answers'}")

    return kmeans, sil_score


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3: SEMI-SUPERVISED - Label Propagation  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def train_label_propagation(X_train, y_train):
    """
    Label Propagation (Semi-Supervised)
    ------------------------------------"""
    print("\n[5/5] Training Label Propagation (Semi-Supervised)...")

    rng     = np.random.RandomState(42)
    indices = rng.choice(X_train.shape[0], size=min(5000, X_train.shape[0]), replace=False)
    X_sub   = X_train[indices].toarray()
    y_sub   = y_train[indices].copy()

    mask_unlabeled         = rng.rand(len(y_sub)) < 0.80
    y_semi                 = y_sub.copy()
    y_semi[mask_unlabeled] = -1

    t0 = time.time()
    lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
    lp.fit(X_sub, y_semi)
    print(f"  Trained in {time.time()-t0:.1f}s")

    labeled_mask = ~mask_unlabeled
    y_true_lp    = y_sub[labeled_mask]
    y_pred_lp    = lp.predict(X_sub[labeled_mask])

    acc_lp = accuracy_score(y_true_lp, y_pred_lp)
    f1_lp  = f1_score(y_true_lp, y_pred_lp, average='macro')
    print(f"  Label Propagation Accuracy : {acc_lp:.4f}")
    print(f"  Label Propagation Macro F1 : {f1_lp:.4f}")

    return lp


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4: WEIGHTED ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

def train_ensemble(lr_model, svm_model, nb_model, X_train, y_train):
    """
    Soft-Vote Ensemble
    -------------------
    """
    print("\n[ENSEMBLE] Building Weighted Soft-Vote Ensemble (LR x2 + SVM x1 + NB x1)...")

    ensemble = VotingClassifier(
        estimators=[
            ('lr',  lr_model),
            ('svm', svm_model),
            ('nb',  nb_model),
        ],
        voting='soft',
        weights=[2, 1, 1],   # LR gets double weight
    )

    t0 = time.time()
    ensemble.fit(X_train, y_train)
    print(f"  Ensemble built in {time.time()-t0:.1f}s")
    return ensemble


# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5: NLP GENERATION METRICS
#
#  Evaluated with BLEU, ROUGE, and METEOR
#
#  How it works:
#  - We take N samples from the dev set that already have RACE gold questions.
#  - We run our template generator on each article to produce a generated question.
#  - We compare generated vs gold using BLEU, ROUGE-L, and METEOR.
# ════════════════════════════════════════════════════════════════════════════

def _template_generate_question(article: str, vectorizer, svm_model) -> str:
    """
    Inline template generator (mirrors inference.py logic) so we can run 
    NLP evaluation inside the training script without importing inference.py.
    """
    STOP = STOPWORDS

    sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return "What is the main idea of the passage?"

    best_sent  = None
    best_score = -1.0

    for sent in sentences:
        tokens      = sent.split()
        chunks, cur = [], []
        for tok in tokens:
            tc = re.sub(r'[^\w]', '', tok).lower()
            if tc and tc not in STOP and len(tc) > 1:
                cur.append(tok)
            else:
                if cur: chunks.append(cur[:])
                cur = []
        if cur: chunks.append(cur[:])
        if not chunks:
            continue
        best_chunk    = max(chunks, key=len)
        answer_phrase = ' '.join(best_chunk[:5])

        try:
            v_s = vectorizer.transform([sent.lower()])
            v_a = vectorizer.transform([answer_phrase.lower()])
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            overlap = float(cos_sim(v_s, v_a)[0][0])
        except Exception:
            overlap = 0.0

        if overlap == 0.0:
            continue

        # Use cosine overlap as a proxy score (no ML ranker needed here)
        if overlap > best_score:
            best_score = overlap
            best_sent  = (sent, answer_phrase)

    if best_sent is None:
        return "What is the main idea of the passage?"

    sent, answer_phrase = best_sent
    sent_lower = sent.lower()
    if any(kw in sent_lower for kw in ['because','reason','therefore']):
        wh = "Why"
    elif any(kw in sent_lower for kw in ['when','year','day','time','century']):
        wh = "When"
    elif any(kw in sent_lower for kw in ['where','place','city','country']):
        wh = "Where"
    elif any(kw in sent_lower for kw in ['who','person','people','man','woman']):
        wh = "Who"
    else:
        wh = "What"

    pattern = re.compile(re.escape(answer_phrase), re.IGNORECASE)
    q_body  = pattern.sub("_____", sent, count=1)
    if "_____" not in q_body:
        return "What is the main idea of the passage?"

    q_body   = re.sub(r'^(the|a|an)\s+', '', q_body.strip(), flags=re.IGNORECASE)
    question = f"{wh} {q_body[0].upper() + q_body[1:]}".rstrip('.!?') + '?'
    return question[:130]


def evaluate_question_generation(dev_df, vectorizer, svm_model, n_samples=200):
    """
    Evaluate the template question generator using BLEU, ROUGE-L, and METEOR.

    These metrics compare our generated questions to the gold RACE questions.
    - BLEU:    n-gram precision; how many of our words appear in the reference.
    - ROUGE-L: longest common subsequence recall.
    - METEOR:  considers synonyms and stemming; more lenient than BLEU.

    A good baseline for template-based generation:
      BLEU  ~0.05-0.15  (neural systems reach ~0.15-0.25)
      ROUGE ~0.20-0.35
      METEOR ~0.10-0.20
    """
    if not NLP_METRICS_AVAILABLE:
        print("  [SKIP] NLP metrics not available. pip install nltk rouge-score")
        return {}

    print(f"\n  Evaluating question generation on {n_samples} dev samples...")

    sample = dev_df.sample(min(n_samples, len(dev_df)), random_state=99)

    smoother    = SmoothingFunction().method1
    rscorer     = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    bleu_scores   = []
    rouge_scores  = []
    meteor_scores = []

    for _, row in sample.iterrows():
        article        = str(row.get('clean_article', ''))
        gold_question  = str(row.get('clean_question', ''))

        if not article.strip() or not gold_question.strip():
            continue

        generated_q = _template_generate_question(article, vectorizer, svm_model)

        ref_tokens = gold_question.lower().split()
        hyp_tokens = generated_q.lower().split()

        # BLEU (up to 2-gram, smoothed to avoid zero for short outputs)
        bleu = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5),
            smoothing_function=smoother
        )
        bleu_scores.append(bleu)

        # ROUGE-L
        rouge_result = rscorer.score(gold_question.lower(), generated_q.lower())
        rouge_scores.append(rouge_result['rougeL'].fmeasure)

        # METEOR
        try:
            met = meteor_score([ref_tokens], hyp_tokens)
        except Exception:
            met = 0.0
        meteor_scores.append(met)

    avg_bleu   = float(np.mean(bleu_scores))   if bleu_scores   else 0.0
    avg_rouge  = float(np.mean(rouge_scores))  if rouge_scores  else 0.0
    avg_meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0

    print(f"\n  {'='*50}")
    print(f"  QUESTION GENERATION EVALUATION  ({len(bleu_scores)} samples)")
    print(f"  {'='*50}")
    print(f"  BLEU-2  Score : {avg_bleu:.4f}")
    print(f"  ROUGE-L Score : {avg_rouge:.4f}")
    print(f"  METEOR  Score : {avg_meteor:.4f}")
    print(f"  {'='*50}")
    print(f"  Interpretation:")
    print(f"    BLEU  measures n-gram precision vs gold questions.")
    print(f"    ROUGE-L measures longest-common-subsequence recall.")
    print(f"    METEOR considers stemming and synonyms.")
    print(f"    Template-based generators typically score BLEU 0.05-0.15.")
    print(f"    These scores reflect text similarity, not question validity.")

    # Save a bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = ['BLEU-2', 'ROUGE-L', 'METEOR']
    scores       = [avg_bleu, avg_rouge, avg_meteor]
    colors       = ['#4C72B0', '#55A868', '#DD8452']
    bars = ax.bar(metric_names, scores, color=colors, alpha=0.85)
    ax.set_ylim(0, max(scores) * 1.4 + 0.01)
    ax.set_ylabel('Score')
    ax.set_title('Model A - Question Generation NLP Metrics', fontweight='bold')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, 'model_a_nlp_metrics.png')
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    print(f"  NLP metrics chart saved -> {chart_path}")

    return {
        "bleu_2"  : round(avg_bleu,   4),
        "rouge_l" : round(avg_rouge,  4),
        "meteor"  : round(avg_meteor, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MODEL A - Answer Verifier Training  (IMPROVED)")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n>>> Loading training data...")
    train_df, vectorizer = load_data("train")
    dev_df,   _          = load_data("dev")

    # ── Build feature matrices ─────────────────────────────────────────────
    train_limit = None if DO_FULL_TRAIN else 20000
    dev_limit   = None if DO_FULL_TRAIN else 5000
    str_temp    = "Full dataset" if DO_FULL_TRAIN else "Small dataset"

    print(f"\n>>> Expanding training set ({str_temp}) with OHE + hand-crafted features...")
    X_train, y_train = expand_to_verification_pairs(train_df, vectorizer, max_rows=train_limit)

    print("\n>>> Expanding dev set...")
    X_dev, y_dev = expand_to_verification_pairs(dev_df, vectorizer, max_rows=dev_limit)

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

    # ── Weighted Ensemble ──────────────────────────────────────────────────
    ensemble_model = train_ensemble(lr_model, svm_model, nb_model, X_train, y_train)
    results.append(evaluate(ensemble_model, X_dev, y_dev, "Weighted Soft-Vote Ensemble"))

    # ── NLP Generation Metrics (BLEU / ROUGE / METEOR) ────────────────────
    print("\n>>> Evaluating Question Generator with BLEU, ROUGE-L, METEOR...")
    nlp_metrics = evaluate_question_generation(dev_df, vectorizer, svm_model, n_samples=200)

    # ── Comparison table ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON TABLE (Classification)")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\n  K-Means Silhouette Score : {sil_score:.4f}")

    if nlp_metrics:
        print("\n" + "=" * 60)
        print("  NLP GENERATION METRICS (Question Generator)")
        print("=" * 60)
        print(f"  BLEU-2  : {nlp_metrics.get('bleu_2',  0):.4f}")
        print(f"  ROUGE-L : {nlp_metrics.get('rouge_l', 0):.4f}")
        print(f"  METEOR  : {nlp_metrics.get('meteor',  0):.4f}")

    # Save results CSV (includes both classification and NLP metrics)
    results_df.to_csv(os.path.join(MODELS_DIR, 'model_a_results.csv'), index=False)

    if nlp_metrics:
        nlp_df = pd.DataFrame([{
            "metric": "BLEU-2",   "score": nlp_metrics.get("bleu_2",  0)
        }, {
            "metric": "ROUGE-L",  "score": nlp_metrics.get("rouge_l", 0)
        }, {
            "metric": "METEOR",   "score": nlp_metrics.get("meteor",  0)
        }])
        nlp_df.to_csv(os.path.join(MODELS_DIR, 'model_a_nlp_metrics.csv'), index=False)
        print(f"  NLP metrics saved -> {os.path.join(MODELS_DIR, 'model_a_nlp_metrics.csv')}")

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
    model_names = [r['model'] for r in results]
    accs        = [float(r['accuracy'])   for r in results]
    f1s         = [float(r['macro_f1'])   for r in results]
    f1c         = [float(r['f1_correct']) for r in results]
    x           = np.arange(len(model_names))
    width       = 0.25
    ax.bar(x - width,     accs, width, label='Accuracy',        color='#4C72B0', alpha=0.85)
    ax.bar(x,             f1s,  width, label='Macro F1',         color='#DD8452', alpha=0.85)
    ax.bar(x + width,     f1c,  width, label='F1 (Correct cls)', color='#55A868', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Model A - Supervised Model Comparison (Improved)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_a_comparison.png'), bbox_inches='tight')
    plt.close()
    print("Comparison chart saved!")

    print("\n[SUCCESS] Model A training complete!")


if __name__ == "__main__":
    main()