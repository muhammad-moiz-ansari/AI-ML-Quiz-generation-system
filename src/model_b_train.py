"""
model_b_train.py
================
Model B - Distractor Generator + Hint Generator
"""

import os
import re
import time
import string
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix, precision_score, recall_score)
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────────────────────────
is_local = False
DO_FULL_EVAL = True   # Set True for full evaluation (slow), False for quick test

if is_local:
    PROCESSED_DIR = "../data/processed/"
    MODELS_DIR_A  = "../models/model_a/traditional/"
    MODELS_DIR_B  = "../models/model_b/traditional/"
    PLOTS_DIR     = "../notebooks/plots/"
else:
    BASE_DIR = "/content/drive/MyDrive/AI PROJECT/AI PROJECT/"
    PROCESSED_DIR = os.path.join(BASE_DIR, "data/processed/")
    MODELS_DIR_A  = os.path.join(BASE_DIR, "models/model_a/traditional/")
    MODELS_DIR_B  = os.path.join(BASE_DIR, "models/model_b/traditional/")
    PLOTS_DIR     = os.path.join(BASE_DIR, "notebooks/plots/")

os.makedirs(MODELS_DIR_B, exist_ok=True)
os.makedirs(PLOTS_DIR,    exist_ok=True)

STOPWORDS = {
    'the','a','an','is','was','are','were','to','of','in','for',
    'on','with','at','by','from','it','its','this','that','and',
    'or','but','not','be','been','has','have','had','he','she',
    'they','we','i','you','do','did','will','would','could','should',
    'as','if','than','then','so','no','up','out','about','into'
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text):
    """Lowercase and remove punctuation."""
    return str(text).lower().translate(str.maketrans('', '', string.punctuation))


def get_keywords(text):
    """Return meaningful words after stopword removal."""
    return set(clean_text(text).split()) - STOPWORDS


def load_data(split="train"):
    df         = pd.read_csv(os.path.join(PROCESSED_DIR, f"clean_{split}.csv"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/onehot_encoder.pkl"))
    return df, vectorizer


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DISTRACTOR GENERATION (Cosine Similarity Pipeline)
# ════════════════════════════════════════════════════════════════════════════

def get_distractor_candidates_cosine(row, vectorizer, n_candidates=10):
    """
    Step 1: Split article into sentences.
    Step 2: Vectorize each sentence + the correct answer.
    Step 3: Rank by cosine similarity to the answer.
    Step 4: Return medium-similarity sentences as distractor candidates
            (high sim = too close to answer, low sim = irrelevant).

    Returns list of (sentence, similarity_score)
    """
    article   = str(row.get('clean_article', ''))
    answer    = str(row.get(f"clean_{row['answer']}", ''))

    # Split into sentences (rough split on punctuation)
    sentences = [s.strip() for s in re.split(r'[.!?]+', article) if len(s.strip().split()) > 3]

    if len(sentences) == 0:
        return []

    # Vectorize
    all_texts  = sentences + [answer]
    X          = vectorizer.transform(all_texts)
    X_sents    = X[:-1]
    X_answer   = X[-1]

    # Cosine similarity of each sentence to the correct answer
    sims = cosine_similarity(X_sents, X_answer).flatten()

    # Target medium similarity: not too close (would give away answer),
    # not too far (would be obviously wrong)
    # Sort by |sim - 0.3| ascending → closest to medium similarity first
    scored = [(sentences[i], sims[i]) for i in range(len(sentences))]
    scored.sort(key=lambda x: abs(x[1] - 0.30))

    return scored[:n_candidates]


def select_top3_distractors(candidates, correct_answer_text, diversity_threshold=0.15):
    """
    From candidates, pick 3 that are:
    1. Not the correct answer (no exact overlap)
    2. Diverse from each other (pairwise word overlap < threshold)
    """
    answer_words = get_keywords(correct_answer_text)
    selected     = []

    for sent, score in candidates:
        sent_words = get_keywords(sent)

        # Skip if too similar to correct answer
        if len(answer_words) > 0:
            overlap = len(answer_words & sent_words) / (len(answer_words) + 1e-9)
            if overlap > 0.7:
                continue

        # Diversity check: not too similar to already selected distractors
        too_similar = False
        for prev in selected:
            prev_words = get_keywords(prev)
            jaccard    = len(sent_words & prev_words) / (len(sent_words | prev_words) + 1e-9)
            if jaccard > diversity_threshold:
                too_similar = True
                break

        if not too_similar:
            selected.append(sent)

        if len(selected) == 3:
            break

    # Pad with fallbacks if fewer than 3 found
    fallbacks = ["Not mentioned in the passage.",
                 "The opposite of what was described.",
                 "None of the above."]
    while len(selected) < 3:
        selected.append(fallbacks[len(selected)])

    return selected


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ML DISTRACTOR RANKER
# ════════════════════════════════════════════════════════════════════════════

def build_distractor_ranker_dataset(df, vectorizer, max_rows=None):
    """
    Build a training set for the distractor ranker.
    For each RACE question, the 3 actual wrong options (B/C/D when A is correct)
    are POSITIVE examples (good distractors). Random sentences from the article
    that are not options are NEGATIVE examples (bad distractors).
    Features per candidate:
        - cosine similarity to correct answer
        - cosine similarity to question
        - word overlap with correct answer (Jaccard)
        - candidate length (word count)
        - position of sentence in article (normalized)
    """
    if max_rows is not None:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    print(f"  Building ranker dataset from {len(df):,} rows...")
    rows_sample = df

    X_feat = []
    y_feat = []

    for _, row in rows_sample.iterrows():
        correct_opt = row['answer']
        opts        = ['A', 'B', 'C', 'D']
        wrong_opts  = [o for o in opts if o != correct_opt]

        answer_text   = str(row.get(f'clean_{correct_opt}', ''))
        question_text = str(row.get('clean_question', ''))
        article_text  = str(row.get('clean_article', ''))
        sentences     = [s.strip() for s in re.split(r'[.!?]+', article_text)
                         if len(s.strip().split()) > 2]

        try:
            v_ans = vectorizer.transform([answer_text])
            v_q   = vectorizer.transform([question_text])
        except Exception:
            continue

        def get_features(candidate_text):
            try:
                v_cand     = vectorizer.transform([candidate_text])
                sim_ans    = cosine_similarity(v_cand, v_ans)[0][0]
                sim_q      = cosine_similarity(v_cand, v_q)[0][0]
                ans_words  = get_keywords(answer_text)
                cand_words = get_keywords(candidate_text)
                jaccard    = (len(ans_words & cand_words) /
                              (len(ans_words | cand_words) + 1e-9))
                length     = min(len(candidate_text.split()) / 50.0, 1.0)
                return [sim_ans, sim_q, jaccard, length]
            except Exception:
                return [0.0, 0.0, 0.0, 0.0]

        # Positive: actual wrong options from RACE
        for opt in wrong_opts:
            opt_text = str(row.get(f'clean_{opt}', ''))
            X_feat.append(get_features(opt_text))
            y_feat.append(1)

        # Negative: random article sentences
        for sent in sentences[:3]:
            X_feat.append(get_features(sent))
            y_feat.append(0)

    return np.array(X_feat), np.array(y_feat)


def train_distractor_ranker(X_feat, y_feat):
    """Train LR and RF rankers on distractor features."""
    print("\n[1/3] Training Distractor Ranker (LR + RF)...")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_feat,
                                               test_size=0.2, random_state=42)

    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    t0 = time.time()
    lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)
    print(f"  Trained in {time.time()-t0:.1f}s")

    results = {}
    for name, model in [("LR Ranker", lr), ("RF Ranker", rf)]:
        y_pred = model.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        f1     = f1_score(y_te, y_pred, average='macro')
        prec   = precision_score(y_te, y_pred, zero_division=0)
        rec    = recall_score(y_te, y_pred, zero_division=0)
        results[name] = {"Accuracy": acc, "Macro F1": f1,
                         "Precision": prec, "Recall": rec}
        print(f"\n  {name}:")
        print(f"    Accuracy  : {acc:.4f}")
        print(f"    Macro F1  : {f1:.4f}")
        print(f"    Precision : {prec:.4f}")
        print(f"    Recall    : {rec:.4f}")
        print(classification_report(y_te, y_pred,
                                    target_names=['Bad Dist.', 'Good Dist.']))

    return lr, rf, results


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HINT GENERATOR
# ════════════════════════════════════════════════════════════════════════════


def build_hint_scorer_dataset(df, vectorizer, max_rows=None):
    """
    For each question, score every sentence in the article.
    Label = 1 if the sentence contains the correct answer text, else 0.
    Features:
        - cosine similarity of sentence to question
        - keyword overlap (Jaccard) with question
        - sentence position in article (0 = first, 1 = last)
        - sentence length normalized
    """
    if max_rows is not None:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    print(f"  Building hint scorer dataset from {len(df):,} rows...")
    rows_sample = df

    X_feat = []
    y_feat = []

    for _, row in rows_sample.iterrows():
        article_text  = str(row.get('clean_article', ''))
        question_text = str(row.get('clean_question', ''))
        answer_text   = str(row.get(f"clean_{row['answer']}", ''))

        sentences = [s.strip() for s in re.split(r'[.!?]+', article_text)
                     if len(s.strip().split()) > 2]
        if len(sentences) == 0:
            continue

        try:
            v_q = vectorizer.transform([question_text])
        except Exception:
            continue

        ans_words = get_keywords(answer_text)
        q_words   = get_keywords(question_text)

        for pos, sent in enumerate(sentences):
            try:
                v_s      = vectorizer.transform([sent])
                sim_q    = cosine_similarity(v_s, v_q)[0][0]
                s_words  = get_keywords(sent)
                jacc_q   = (len(q_words & s_words) /
                            (len(q_words | s_words) + 1e-9))
                norm_pos = pos / (len(sentences) - 1 + 1e-9)
                norm_len = min(len(sent.split()) / 40.0, 1.0)

                X_feat.append([sim_q, jacc_q, norm_pos, norm_len])

                ans_overlap = len(ans_words & s_words) / (len(ans_words) + 1e-9)
                y_feat.append(1 if ans_overlap > 0.4 else 0)

            except Exception:
                continue

    return np.array(X_feat), np.array(y_feat)

def train_hint_scorer(X_feat, y_feat):
    """Train LR hint scorer on sentence features."""
    print("\n[2/3] Training Hint Scorer (Logistic Regression)...")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_feat,
                                               test_size=0.2, random_state=42)

    model = LogisticRegression(C=1.0, max_iter=500,
                               class_weight='balanced', random_state=42)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    print(f"  Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1     = f1_score(y_te, y_pred, average='macro')
    prec   = precision_score(y_te, y_pred, zero_division=0)
    rec    = recall_score(y_te, y_pred, zero_division=0)

    print(f"  Hint Scorer Accuracy  : {acc:.4f}")
    print(f"  Hint Scorer Macro F1  : {f1:.4f}")
    print(f"  Hint Scorer Precision : {prec:.4f}")
    print(f"  Hint Scorer Recall    : {rec:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=['Not Hint', 'Hint']))

    return model, {"Accuracy": acc, "Macro F1": f1,
                   "Precision": prec, "Recall": rec}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INFERENCE FUNCTIONS (used by Streamlit UI)
# ════════════════════════════════════════════════════════════════════════════

def generate_distractors(article, question, correct_answer,
                         vectorizer, lr_ranker):
    """
    Full distractor generation pipeline for a single sample.
    Returns list of 3 distractor strings.
    """
    row_like = {
        'clean_article':  clean_text(article),
        'clean_question': clean_text(question),
        'answer':         'A',
        'clean_A':        clean_text(correct_answer),
    }

    candidates = get_distractor_candidates_cosine(row_like, vectorizer, n_candidates=15)
    distractors = select_top3_distractors(candidates, correct_answer)
    return distractors


def generate_hints(article, question, correct_answer,
                   vectorizer, hint_scorer):
    """
    Generate 3 graduated hints for a single sample.
    Hint 1 = vaguest, Hint 3 = most explicit (nearest to answer).
    """
    article_clean  = clean_text(article)
    question_clean = clean_text(question)

    sentences = [s.strip() for s in re.split(r'[.!?]+', article_clean)
                 if len(s.strip().split()) > 3]
    if len(sentences) == 0:
        return ["Re-read the passage carefully.",
                "Focus on the key events described.",
                "Look for the part that directly answers the question."]

    try:
        v_q = vectorizer.transform([question_clean])
        q_words = get_keywords(question_clean)

        features = []
        for pos, sent in enumerate(sentences):
            v_s     = vectorizer.transform([sent])
            sim_q   = cosine_similarity(v_s, v_q)[0][0]
            s_words = get_keywords(sent)
            jacc_q  = len(q_words & s_words) / (len(q_words | s_words) + 1e-9)
            norm_pos = pos / (len(sentences) - 1 + 1e-9)
            norm_len = min(len(sent.split()) / 40.0, 1.0)
            features.append([sim_q, jacc_q, norm_pos, norm_len])

        probs  = hint_scorer.predict_proba(np.array(features))[:, 1]
        ranked = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)

        # Pick top 3 diverse hints
        hints = []
        for sent, prob in ranked:
            if all(len(get_keywords(sent) & get_keywords(h)) /
                   (len(get_keywords(sent) | get_keywords(h)) + 1e-9) < 0.5
                   for h in hints):
                hints.append(sent)
            if len(hints) == 3:
                break

        # Pad if needed
        fallbacks = ["Re-read the passage carefully.",
                     "Focus on the key events described.",
                     "Look for the part that directly answers the question."]
        while len(hints) < 3:
            hints.append(fallbacks[len(hints)])

        # Hint 1 = vaguest (lowest prob), Hint 3 = most explicit (highest)
        hints.reverse()
        return hints

    except Exception as e:
        print(f"  Hint generation error: {e}")
        return ["Re-read the passage carefully.",
                "Focus on the key events described.",
                "Look for the part that directly answers the question."]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION ON DEV SET
# ════════════════════════════════════════════════════════════════════════════

def evaluate_distractor_pipeline(df, vectorizer, lr_ranker, n_samples=200):
    """
    On n_samples from dev set:
    - Generate 3 distractors per question
    - Check how many of the 3 real wrong options we recovered (Recall)
    - Check how many of our generated ones appear in real options (Precision)
    - Accuracy = fraction where top distractor is NOT the correct answer
    """
    print(f"\n[3/3] Evaluating distractor pipeline on {n_samples} samples...")
    sample = df.sample(min(n_samples, len(df)), random_state=99)

    precision_scores = []
    recall_scores    = []
    accuracy_scores  = []
    diversity_scores = []

    for _, row in sample.iterrows():
        correct_opt  = row['answer']
        wrong_opts   = [o for o in ['A','B','C','D'] if o != correct_opt]
        gold_distractors = set(
            clean_text(str(row.get(f'clean_{o}', ''))) for o in wrong_opts
        )
        correct_ans_text = str(row.get(f'clean_{correct_opt}', ''))

        row_like = {
            'clean_article':  str(row.get('clean_article', '')),
            'clean_question': str(row.get('clean_question', '')),
            'answer': correct_opt,
            f'clean_{correct_opt}': correct_ans_text
        }

        candidates   = get_distractor_candidates_cosine(row_like, vectorizer, n_candidates=15)
        gen_dist     = select_top3_distractors(candidates, correct_ans_text)
        gen_dist_set = set(clean_text(d) for d in gen_dist)

        # Precision: overlap with gold distractors
        hits = len(gen_dist_set & gold_distractors)
        precision_scores.append(hits / (len(gen_dist_set) + 1e-9))
        recall_scores.append(hits / (len(gold_distractors) + 1e-9))

        # Accuracy: none of our distractors is the correct answer
        correct_in_gen = any(
            len(get_keywords(correct_ans_text) & get_keywords(d)) /
            (len(get_keywords(correct_ans_text) | get_keywords(d)) + 1e-9) > 0.8
            for d in gen_dist
        )
        accuracy_scores.append(0 if correct_in_gen else 1)

        # Diversity: average pairwise Jaccard distance
        pairs = [(gen_dist[i], gen_dist[j])
                 for i in range(len(gen_dist))
                 for j in range(i+1, len(gen_dist))]
        if pairs:
            div = np.mean([
                1 - len(get_keywords(a) & get_keywords(b)) /
                    (len(get_keywords(a) | get_keywords(b)) + 1e-9)
                for a, b in pairs
            ])
            diversity_scores.append(div)

    prec = np.mean(precision_scores)
    rec  = np.mean(recall_scores)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = np.mean(accuracy_scores)
    div  = np.mean(diversity_scores) if diversity_scores else 0.0

    print(f"\n  Distractor Pipeline Results ({n_samples} samples):")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  F1                 : {f1:.4f}")
    print(f"  Accuracy (no leak) : {acc:.4f}")
    print(f"  Diversity (Jaccard): {div:.4f}")

    return {"Precision": prec, "Recall": rec, "F1": f1,
            "Accuracy": acc, "Diversity": div}


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_model_b_results(dist_results, hint_results, dist_ranker_results):
    metrics_names = ['Precision', 'Recall', 'F1', 'Accuracy']

    # Chart 1: Distractor pipeline metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [dist_results.get(m, 0) for m in metrics_names]
    ax.bar(metrics_names, vals, color=['#4C72B0','#DD8452','#55A868','#C44E52'], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title('Model B - Distractor Pipeline Evaluation', fontweight='bold')
    ax.set_ylabel('Score')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_b_distractor_eval.png'), bbox_inches='tight')
    plt.close()

    # Chart 2: Ranker comparison
    names  = list(dist_ranker_results.keys())
    accs   = [dist_ranker_results[n]['Accuracy'] for n in names]
    f1s    = [dist_ranker_results[n]['Macro F1'] for n in names]
    x      = np.arange(len(names))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#DD8452', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title('Model B - Distractor Ranker Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_b_ranker_comparison.png'), bbox_inches='tight')
    plt.close()

    print("  Plots saved!")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MODEL B - Distractor & Hint Generator Training")
    print("=" * 60)

    # ── Load Data ──────────────────────────────────────────────────────────
    print("\n>>> Loading data...")
    train_df, vectorizer = load_data("train")
    dev_df,   _          = load_data("dev")

    ranker_rows = None if DO_FULL_EVAL else 3000
    hint_rows   = None if DO_FULL_EVAL else 3000
    eval_rows   = 500  if DO_FULL_EVAL else 100

    # ── Distractor Ranker ─────────────────────────────────────────────────
    print("\n>>> Building Distractor Ranker dataset...")
    X_dist, y_dist = build_distractor_ranker_dataset(
        train_df, vectorizer,
        max_rows=ranker_rows
    )
    print(f"  Dataset: {X_dist.shape[0]:,} examples  |  features: {X_dist.shape[1]}")
    print(f"  Class balance: {y_dist.mean():.2f} positive")

    lr_ranker, rf_ranker, ranker_results = train_distractor_ranker(X_dist, y_dist)

    # ── Hint Scorer ───────────────────────────────────────────────────────
    print("\n>>> Building Hint Scorer dataset...")
    X_hint, y_hint = build_hint_scorer_dataset(
        train_df, vectorizer,
        max_rows=hint_rows
    )
    print(f"  Dataset: {X_hint.shape[0]:,} examples  |  features: {X_hint.shape[1]}")
    print(f"  Class balance: {y_hint.mean():.2f} positive (hint sentences)")

    hint_model, hint_results = train_hint_scorer(X_hint, y_hint)

    # ── Pipeline Evaluation on Dev ────────────────────────────────────────
    dist_pipeline_results = evaluate_distractor_pipeline(
        dev_df, vectorizer, lr_ranker, n_samples=eval_rows
    )

    # ── Quick Demo ────────────────────────────────────────────────────────
    print("\n>>> Running quick demo on 1 sample from dev set...")
    sample_row  = dev_df.iloc[0]
    correct_opt = sample_row['answer']

    try:
        raw_dev   = pd.read_csv(os.path.join(BASE_DIR, "data/raw/dev.csv"))
        raw_row   = raw_dev.iloc[0]
        article   = str(raw_row['article'])
        question  = str(raw_row['question'])
        correct_answer_text = str(raw_row[correct_opt])
    except Exception:
        article   = str(sample_row.get('clean_article', ''))
        question  = str(sample_row.get('clean_question', ''))
        correct_answer_text = str(sample_row.get(f'clean_{correct_opt}', ''))

    print(f"\n  Question : {question[:100]}...")
    print(f"  Answer   : {correct_answer_text}")

    distractors = generate_distractors(article, question, correct_answer_text,
                                       vectorizer, lr_ranker)
    hints       = generate_hints(article, question, correct_answer_text,
                                 vectorizer, hint_model)

    print("\n  Generated Distractors:")
    for i, d in enumerate(distractors, 1):
        print(f"    Distractor {i}: {d[:100]}")

    print("\n  Generated Hints (1=vague, 3=explicit):")
    for i, h in enumerate(hints, 1):
        print(f"    Hint {i}: {h[:120]}")

    # ── Save Models ───────────────────────────────────────────────────────
    print("\n>>> Saving Model B...")
    joblib.dump(lr_ranker,  os.path.join(MODELS_DIR_B, 'distractor_lr_ranker.pkl'))
    joblib.dump(rf_ranker,  os.path.join(MODELS_DIR_B, 'distractor_rf_ranker.pkl'))
    joblib.dump(hint_model, os.path.join(MODELS_DIR_B, 'hint_scorer.pkl'))
    print("  All models saved!")

    # ── Save Results ──────────────────────────────────────────────────────
    results_rows = []
    for name, res in ranker_results.items():
        results_rows.append({"component": f"Distractor Ranker ({name})", **res})
    results_rows.append({"component": "Hint Scorer", **hint_results})
    results_rows.append({"component": "Distractor Pipeline", **dist_pipeline_results})
    pd.DataFrame(results_rows).to_csv(
        os.path.join(MODELS_DIR_B, 'model_b_results.csv'), index=False
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_model_b_results(dist_pipeline_results, hint_results, ranker_results)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  MODEL B - FINAL RESULTS")
    print("═" * 60)
    print(f"  Distractor Ranker (LR) Accuracy  : {ranker_results['LR Ranker']['Accuracy']:.4f}")
    print(f"  Distractor Ranker (RF) Accuracy  : {ranker_results['RF Ranker']['Accuracy']:.4f}")
    print(f"  Hint Scorer Accuracy             : {hint_results['Accuracy']:.4f}")
    print(f"  Distractor Pipeline Precision    : {dist_pipeline_results['Precision']:.4f}")
    print(f"  Distractor Pipeline Recall       : {dist_pipeline_results['Recall']:.4f}")
    print(f"  Distractor Pipeline F1           : {dist_pipeline_results['F1']:.4f}")
    print(f"  Distractor Diversity             : {dist_pipeline_results['Diversity']:.4f}")
    print("\n✅ Model B training complete!")


if __name__ == "__main__":
    main()