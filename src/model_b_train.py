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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              precision_score, recall_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.class_weight import compute_sample_weight

# ── Paths ────────────────────────────────────────────────────────────────────
is_local = False
DO_FULL_EVAL = True

if is_local:
    PROCESSED_DIR = "../data/processed/"
    MODELS_DIR_A  = "../models/model_a/traditional/"
    MODELS_DIR_B  = "../models/model_b/traditional/"
    PLOTS_DIR     = "../notebooks/plots/"
else:
    BASE_DIR      = "/content/drive/MyDrive/Colab Notebooks/AI PROJECT/"
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
    return str(text).lower().translate(str.maketrans('', '', string.punctuation))


def get_keywords(text):
    return set(clean_text(text).split()) - STOPWORDS


def load_data(split="train"):
    df         = pd.read_csv(os.path.join(PROCESSED_DIR, f"clean_{split}.csv"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "models/onehot_encoder.pkl"))
    return df, vectorizer


def extract_phrase_chunks(text, chunk_size=5):
    """
    Splits text into overlapping phrase-length chunks (similar length to
    RACE answer options) so negative distractor examples match the
    length distribution of positive examples.
    This is the key fix for the data leakage problem.
    """
    tokens = text.split()
    chunks = []
    step   = max(1, chunk_size // 2)   # 50% overlap
    for i in range(0, max(1, len(tokens) - chunk_size + 1), step):
        chunk = ' '.join(tokens[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 - DISTRACTOR PIPELINE (Cosine Similarity)
# ════════════════════════════════════════════════════════════════════════════

def get_distractor_candidates_cosine(row, vectorizer, n_candidates=10):
    """
    Extract phrase-length distractor candidates from the article.
    Uses phrase chunks (same length as answer options) rather than full
    sentences to produce cleaner, more realistic distractors.
    """
    article = str(row.get('clean_article', ''))
    answer  = str(row.get(f"clean_{row['answer']}", ''))

    # Determine answer length to match chunk size
    ans_len    = max(3, min(len(answer.split()), 8))
    candidates = extract_phrase_chunks(article, chunk_size=ans_len)

    if not candidates:
        return []

    all_texts = candidates + [answer]
    X         = vectorizer.transform(all_texts)
    X_cands   = X[:-1]
    X_answer  = X[-1]

    sims   = cosine_similarity(X_cands, X_answer).flatten()
    scored = [(candidates[i], sims[i]) for i in range(len(candidates))]
    # Target medium similarity ~0.25 (plausible but not the answer)
    scored.sort(key=lambda x: abs(x[1] - 0.25))

    return scored[:n_candidates]


def select_top3_distractors(candidates, correct_answer_text, diversity_threshold=0.15):
    """
    Pick 3 diverse distractors that are not the correct answer.
    """
    answer_words = get_keywords(correct_answer_text)
    selected     = []

    for sent, score in candidates:
        sent_words = get_keywords(sent)

        # Skip if too similar to the correct answer
        if answer_words:
            overlap = len(answer_words & sent_words) / (len(answer_words) + 1e-9)
            if overlap > 0.7:
                continue

        # Diversity check
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

    fallbacks = ["Not mentioned in the passage.",
                 "The opposite of what was described.",
                 "None of the above."]
    while len(selected) < 3:
        selected.append(fallbacks[len(selected)])

    return selected


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 - ML DISTRACTOR RANKER
# ════════════════════════════════════════════════════════════════════════════

def build_distractor_ranker_dataset(df, vectorizer, max_rows=None):
    """
    Build training set for the distractor ranker.

    POSITIVES: actual RACE wrong options (known good distractors).
    NEGATIVES: phrase-length chunks from article that do NOT overlap much
               with any option - same length as positives, fixing data leakage.

    6 Features per candidate:
      0  cosine similarity to correct answer
      1  cosine similarity to question
      2  Jaccard keyword overlap with correct answer
      3  Jaccard keyword overlap with question
      4  normalised candidate length
      5  binary: candidate shares any keyword with correct answer
    """
    if max_rows is not None:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    print(f"  Building ranker dataset from {len(df):,} rows...")

    X_feat = []
    y_feat = []

    for _, row in df.iterrows():
        correct_opt   = row['answer']
        opts          = ['A', 'B', 'C', 'D']
        wrong_opts    = [o for o in opts if o != correct_opt]

        answer_text   = str(row.get(f'clean_{correct_opt}', ''))
        question_text = str(row.get('clean_question', ''))
        article_text  = str(row.get('clean_article', ''))

        # All option texts (for building negative pool)
        all_opt_texts = set(
            clean_text(str(row.get(f'clean_{o}', ''))) for o in opts
        )

        try:
            v_ans = vectorizer.transform([answer_text])
            v_q   = vectorizer.transform([question_text])
        except Exception:
            continue

        ans_words = get_keywords(answer_text)
        q_words   = get_keywords(question_text)

        def get_features(candidate_text):
            try:
                v_cand    = vectorizer.transform([candidate_text])
                sim_ans   = float(cosine_similarity(v_cand, v_ans)[0][0])
                sim_q     = float(cosine_similarity(v_cand, v_q)[0][0])
                cand_words = get_keywords(candidate_text)
                jacc_ans  = len(ans_words & cand_words) / (len(ans_words | cand_words) + 1e-9)
                jacc_q    = len(q_words   & cand_words) / (len(q_words   | cand_words) + 1e-9)
                length    = min(len(candidate_text.split()) / 15.0, 1.0)  # norm by 15 (phrase length)
                shares_kw = 1.0 if (ans_words & cand_words) else 0.0
                return [sim_ans, sim_q, jacc_ans, jacc_q, length, shares_kw]
            except Exception:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ── POSITIVES: actual RACE wrong options ────────────────────────
        for opt in wrong_opts:
            opt_text = str(row.get(f'clean_{opt}', ''))
            if opt_text.strip():
                X_feat.append(get_features(opt_text))
                y_feat.append(1)

        # ── NEGATIVES: phrase chunks from article (SAME length as options)
        # Skip any chunk that closely matches an actual option
        ans_len  = max(3, min(len(answer_text.split()), 8))
        chunks   = extract_phrase_chunks(article_text, chunk_size=ans_len)
        neg_added = 0
        for chunk in chunks:
            chunk_clean = clean_text(chunk)
            # Reject if too similar to any option (would be a true positive)
            skip = False
            for opt_t in all_opt_texts:
                kw_c = get_keywords(chunk_clean)
                kw_o = get_keywords(opt_t)
                if kw_o and len(kw_c & kw_o) / (len(kw_o) + 1e-9) > 0.6:
                    skip = True
                    break
            if not skip:
                X_feat.append(get_features(chunk_clean))
                y_feat.append(0)
                neg_added += 1
            if neg_added >= 3:   # 3 negatives per positive group (3 wrong opts)
                break

    X = np.array(X_feat, dtype=np.float32)
    y = np.array(y_feat, dtype=int)
    return X, y


def train_distractor_ranker(X_feat, y_feat):
    """Train LR and RF rankers on the improved 6-feature distractor dataset."""
    print("\n[1/3] Training Distractor Ranker (LR + RF)...")
    print(f"  Dataset shape: {X_feat.shape}  |  class balance: {y_feat.mean():.2f} positive")

    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_feat,
                                               test_size=0.2, random_state=42,
                                               stratify=y_feat)

    lr = LogisticRegression(C=1.0, max_iter=500,
                            class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                class_weight='balanced',
                                random_state=42, n_jobs=-1)

    t0 = time.time()
    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
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
# SECTION 3 - HINT SCORER
# ════════════════════════════════════════════════════════════════════════════

def build_hint_scorer_dataset(df, vectorizer, max_rows=None):
    """
    Build training set for the hint scorer.

    6 Features per sentence (was 4):
      0  cosine similarity of sentence to question
      1  Jaccard keyword overlap with question
      2  sentence position in article (normalised)
      3  sentence length (normalised)
      4  Jaccard keyword overlap with correct answer  [NEW]
      5  binary: does sentence contain any answer keyword?  [NEW]

    Feature 4 and 5 directly answer "does this sentence reveal the answer?"
    which is exactly what a good hint should do.
    """
    if max_rows is not None:
        df = df.sample(min(max_rows, len(df)), random_state=42)
    print(f"  Building hint scorer dataset from {len(df):,} rows...")

    X_feat = []
    y_feat = []

    for _, row in df.iterrows():
        article_text  = str(row.get('clean_article', ''))
        question_text = str(row.get('clean_question', ''))
        answer_text   = str(row.get(f"clean_{row['answer']}", ''))

        sentences = [s.strip() for s in re.split(r'[.!?]+', article_text)
                     if len(s.strip().split()) > 2]
        if not sentences:
            continue

        try:
            v_q = vectorizer.transform([question_text])
        except Exception:
            continue

        ans_words = get_keywords(answer_text)
        q_words   = get_keywords(question_text)

        for pos, sent in enumerate(sentences):
            try:
                v_s       = vectorizer.transform([sent])
                s_words   = get_keywords(sent)

                sim_q     = float(cosine_similarity(v_s, v_q)[0][0])
                jacc_q    = len(q_words & s_words) / (len(q_words | s_words) + 1e-9)
                norm_pos  = pos / (len(sentences) - 1 + 1e-9)
                norm_len  = min(len(sent.split()) / 40.0, 1.0)

                # NEW: overlap with answer
                jacc_ans  = len(ans_words & s_words) / (len(ans_words | s_words) + 1e-9) if ans_words else 0.0
                has_ans_kw = 1.0 if (ans_words and ans_words & s_words) else 0.0

                X_feat.append([sim_q, jacc_q, norm_pos, norm_len, jacc_ans, has_ans_kw])

                # Label: sentence is a good hint if it contains >40% of answer keywords
                ans_overlap = len(ans_words & s_words) / (len(ans_words) + 1e-9) if ans_words else 0.0
                y_feat.append(1 if ans_overlap > 0.4 else 0)

            except Exception:
                continue

    return np.array(X_feat, dtype=np.float32), np.array(y_feat, dtype=int)


def train_hint_scorer(X_feat, y_feat):
    """Train LR hint scorer on the improved 6-feature dataset."""
    print("\n[2/3] Training Hint Scorer (Logistic Regression, 6 features)...")
    print(f"  Dataset shape: {X_feat.shape}  |  class balance: {y_feat.mean():.2f} positive")

    X_tr, X_te, y_tr, y_te = train_test_split(X_feat, y_feat,
                                               test_size=0.2, random_state=42,
                                               stratify=y_feat)

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
    print(classification_report(y_te, y_pred, target_names=['Not Hint', 'Hint']))

    return model, {"Accuracy": acc, "Macro F1": f1, "Precision": prec, "Recall": rec}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 - INFERENCE FUNCTIONS  (unchanged interface)
# ════════════════════════════════════════════════════════════════════════════

def generate_distractors(article, question, correct_answer, vectorizer, lr_ranker):
    row_like = {
        'clean_article':  clean_text(article),
        'clean_question': clean_text(question),
        'answer':         'A',
        'clean_A':        clean_text(correct_answer),
    }
    candidates  = get_distractor_candidates_cosine(row_like, vectorizer, n_candidates=15)
    distractors = select_top3_distractors(candidates, correct_answer)
    return distractors


def generate_hints(article, question, correct_answer, vectorizer, hint_scorer):
    """
    Generate 3 graduated hints using the improved 6-feature hint scorer.
    """
    article_clean  = clean_text(article)
    question_clean = clean_text(question)
    answer_clean   = clean_text(correct_answer)

    sentences = [s.strip() for s in re.split(r'[.!?]+', article_clean)
                 if len(s.strip().split()) > 3]
    if not sentences:
        return ["Re-read the passage carefully.",
                "Focus on the key events described.",
                "Look for the part that directly answers the question."]

    try:
        v_q      = vectorizer.transform([question_clean])
        q_words  = get_keywords(question_clean)
        ans_words = get_keywords(answer_clean)

        features = []
        for pos, sent in enumerate(sentences):
            v_s       = vectorizer.transform([sent])
            s_words   = get_keywords(sent)
            sim_q     = float(cosine_similarity(v_s, v_q)[0][0])
            jacc_q    = len(q_words & s_words) / (len(q_words | s_words) + 1e-9)
            norm_pos  = pos / (len(sentences) - 1 + 1e-9)
            norm_len  = min(len(sent.split()) / 40.0, 1.0)
            jacc_ans  = len(ans_words & s_words) / (len(ans_words | s_words) + 1e-9) if ans_words else 0.0
            has_ans_kw = 1.0 if (ans_words and ans_words & s_words) else 0.0
            features.append([sim_q, jacc_q, norm_pos, norm_len, jacc_ans, has_ans_kw])

        probs  = hint_scorer.predict_proba(np.array(features, dtype=np.float32))[:, 1]
        ranked = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)

        hints = []
        for sent, prob in ranked:
            if all(
                len(get_keywords(sent) & get_keywords(h)) /
                (len(get_keywords(sent) | get_keywords(h)) + 1e-9) < 0.5
                for h in hints
            ):
                hints.append(sent)
            if len(hints) == 3:
                break

        fallbacks = ["Re-read the passage carefully.",
                     "Focus on the key events described.",
                     "Look for the part that directly answers the question."]
        while len(hints) < 3:
            hints.append(fallbacks[len(hints)])

        hints.reverse()   # Hint 1 = vaguest, Hint 3 = most explicit
        return hints

    except Exception as e:
        print(f"  Hint generation error: {e}")
        return ["Re-read the passage carefully.",
                "Focus on the key events described.",
                "Look for the part that directly answers the question."]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 - PIPELINE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_distractor_pipeline(df, vectorizer, lr_ranker, n_samples=200):
    """
    Evaluate the full distractor pipeline.

    Uses PARTIAL keyword Jaccard overlap (threshold 0.3):
    a generated distractor counts as a hit if it shares 30% of keywords
    with a gold distractor.  This is the standard approach for evaluating
    open-ended text generation when exact match is too strict.
    """
    print(f"\n[3/3] Evaluating distractor pipeline on {n_samples} samples...")
    print("  Using partial keyword Jaccard overlap (threshold=0.30) for matching.")
    sample = df.sample(min(n_samples, len(df)), random_state=99)

    precision_scores = []
    recall_scores    = []
    accuracy_scores  = []
    diversity_scores = []

    MATCH_THRESHOLD = 0.30   # generated distractor counts as hit if >=30% keyword overlap

    for _, row in sample.iterrows():
        correct_opt      = row['answer']
        wrong_opts       = [o for o in ['A','B','C','D'] if o != correct_opt]
        gold_distractors = [
            get_keywords(str(row.get(f'clean_{o}', ''))) for o in wrong_opts
        ]
        gold_distractors = [g for g in gold_distractors if g]  # remove empty sets
        correct_ans_text = str(row.get(f'clean_{correct_opt}', ''))

        row_like = {
            'clean_article':  str(row.get('clean_article', '')),
            'clean_question': str(row.get('clean_question', '')),
            'answer':         correct_opt,
            f'clean_{correct_opt}': correct_ans_text,
        }

        candidates = get_distractor_candidates_cosine(row_like, vectorizer, n_candidates=15)
        gen_dist   = select_top3_distractors(candidates, correct_ans_text)

        if not gen_dist or not gold_distractors:
            continue

        # ── Precision: how many generated distractors match a gold one ────
        gen_hits = 0
        for gd in gen_dist:
            gd_kw = get_keywords(gd)
            for gold_kw in gold_distractors:
                if not gd_kw or not gold_kw:
                    continue
                jaccard = len(gd_kw & gold_kw) / (len(gd_kw | gold_kw) + 1e-9)
                if jaccard >= MATCH_THRESHOLD:
                    gen_hits += 1
                    break
        precision_scores.append(gen_hits / (len(gen_dist) + 1e-9))

        # ── Recall: how many gold distractors were covered ────────────────
        gold_hits = 0
        for gold_kw in gold_distractors:
            for gd in gen_dist:
                gd_kw = get_keywords(gd)
                if not gd_kw or not gold_kw:
                    continue
                jaccard = len(gd_kw & gold_kw) / (len(gd_kw | gold_kw) + 1e-9)
                if jaccard >= MATCH_THRESHOLD:
                    gold_hits += 1
                    break
        recall_scores.append(gold_hits / (len(gold_distractors) + 1e-9))

        # ── Accuracy: none of our distractors IS the correct answer ───────
        correct_kw   = get_keywords(correct_ans_text)
        correct_leak = any(
            correct_kw and get_keywords(d) and
            len(correct_kw & get_keywords(d)) / (len(correct_kw | get_keywords(d)) + 1e-9) > 0.8
            for d in gen_dist
        )
        accuracy_scores.append(0.0 if correct_leak else 1.0)

        # ── Diversity: average pairwise Jaccard DISTANCE ──────────────────
        pairs = [(gen_dist[i], gen_dist[j])
                 for i in range(len(gen_dist))
                 for j in range(i + 1, len(gen_dist))]
        if pairs:
            div = np.mean([
                1 - len(get_keywords(a) & get_keywords(b)) /
                    (len(get_keywords(a) | get_keywords(b)) + 1e-9)
                for a, b in pairs
            ])
            diversity_scores.append(div)

    prec = float(np.mean(precision_scores)) if precision_scores else 0.0
    rec  = float(np.mean(recall_scores))    if recall_scores    else 0.0
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = float(np.mean(accuracy_scores))  if accuracy_scores  else 0.0
    div  = float(np.mean(diversity_scores)) if diversity_scores else 0.0

    print(f"\n  Distractor Pipeline Results ({n_samples} samples):")
    print(f"  Precision          : {prec:.4f}  (partial keyword match >= 0.30)")
    print(f"  Recall             : {rec:.4f}")
    print(f"  F1                 : {f1:.4f}")
    print(f"  Accuracy (no leak) : {acc:.4f}")
    print(f"  Diversity (Jaccard): {div:.4f}")

    return {"Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc, "Diversity": div}


# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_model_b_results(dist_results, hint_results, dist_ranker_results):
    metrics_names = ['Precision', 'Recall', 'F1', 'Accuracy']

    # Chart 1: Distractor pipeline
    fig, ax = plt.subplots(figsize=(8, 4))
    vals    = [dist_results.get(m, 0) for m in metrics_names]
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    bars    = ax.bar(metrics_names, vals, color=colors, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title('Model B - Distractor Pipeline Evaluation', fontweight='bold')
    ax.set_ylabel('Score')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_b_distractor_eval.png'), bbox_inches='tight')
    plt.close()

    # Chart 2: Ranker LR vs RF comparison
    names  = list(dist_ranker_results.keys())
    accs   = [dist_ranker_results[n]['Accuracy'] for n in names]
    f1s    = [dist_ranker_results[n]['Macro F1'] for n in names]
    x      = np.arange(len(names))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0', alpha=0.85)
    ax.bar(x + width/2, f1s,  width, label='Macro F1',  color='#DD8452', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title('Model B - Distractor Ranker Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_b_ranker_comparison.png'), bbox_inches='tight')
    plt.close()

    # Chart 3: Hint scorer metrics
    hint_metrics = ['Accuracy', 'Macro F1', 'Precision', 'Recall']
    hint_vals    = [hint_results.get(m, 0) for m in hint_metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(hint_metrics, hint_vals,
                  color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title('Model B - Hint Scorer Evaluation', fontweight='bold')
    ax.set_ylabel('Score')
    for bar, v in zip(bars, hint_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_b_hint_scorer.png'), bbox_inches='tight')
    plt.close()

    print("  Plots saved!")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  MODEL B - Distractor & Hint Generator Training")
    print("=" * 60)

    print("\n>>> Loading data...")
    train_df, vectorizer = load_data("train")
    dev_df,   _          = load_data("dev")

    ranker_rows = None if DO_FULL_EVAL else 3000
    hint_rows   = None if DO_FULL_EVAL else 3000
    eval_rows   = 500  if DO_FULL_EVAL else 100

    # ── Distractor Ranker ─────────────────────────────────────────────────
    print("\n>>> Building Distractor Ranker dataset (phrase-level negatives)...")
    X_dist, y_dist = build_distractor_ranker_dataset(train_df, vectorizer,
                                                      max_rows=ranker_rows)
    print(f"  Dataset: {X_dist.shape[0]:,} examples  |  {X_dist.shape[1]} features")
    print(f"  Class balance: {y_dist.mean():.2f} positive")

    lr_ranker, rf_ranker, ranker_results = train_distractor_ranker(X_dist, y_dist)

    # ── Hint Scorer ───────────────────────────────────────────────────────
    print("\n>>> Building Hint Scorer dataset (6 features)...")
    X_hint, y_hint = build_hint_scorer_dataset(train_df, vectorizer,
                                                max_rows=hint_rows)
    print(f"  Dataset: {X_hint.shape[0]:,} examples  |  {X_hint.shape[1]} features")
    print(f"  Class balance: {y_hint.mean():.2f} positive (hint sentences)")

    hint_model, hint_results = train_hint_scorer(X_hint, y_hint)

    # ── Pipeline Evaluation ───────────────────────────────────────────────
    dist_pipeline_results = evaluate_distractor_pipeline(
        dev_df, vectorizer, lr_ranker, n_samples=eval_rows
    )

    # ── Quick Demo ────────────────────────────────────────────────────────
    print("\n>>> Quick demo on 1 dev sample...")
    sample_row  = dev_df.iloc[0]
    correct_opt = sample_row['answer']
    try:
        raw_dev             = pd.read_csv(os.path.join(BASE_DIR, "data/raw/dev.csv"))
        raw_row             = raw_dev.iloc[0]
        article             = str(raw_row['article'])
        question            = str(raw_row['question'])
        correct_answer_text = str(raw_row[correct_opt])
    except Exception:
        article             = str(sample_row.get('clean_article', ''))
        question            = str(sample_row.get('clean_question', ''))
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

    # ── Save Results CSV ──────────────────────────────────────────────────
    results_rows = []
    for name, res in ranker_results.items():
        results_rows.append({"component": f"Distractor Ranker ({name})", **res})
    results_rows.append({"component": "Hint Scorer",          **hint_results})
    results_rows.append({"component": "Distractor Pipeline",  **dist_pipeline_results})
    pd.DataFrame(results_rows).to_csv(
        os.path.join(MODELS_DIR_B, 'model_b_results.csv'), index=False
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_model_b_results(dist_pipeline_results, hint_results, ranker_results)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL B - FINAL RESULTS")
    print("=" * 60)
    print(f"  Distractor Ranker (LR) Accuracy  : {ranker_results['LR Ranker']['Accuracy']:.4f}")
    print(f"  Distractor Ranker (LR) Macro F1  : {ranker_results['LR Ranker']['Macro F1']:.4f}")
    print(f"  Distractor Ranker (RF) Accuracy  : {ranker_results['RF Ranker']['Accuracy']:.4f}")
    print(f"  Distractor Ranker (RF) Macro F1  : {ranker_results['RF Ranker']['Macro F1']:.4f}")
    print(f"  Hint Scorer Accuracy             : {hint_results['Accuracy']:.4f}")
    print(f"  Hint Scorer Macro F1             : {hint_results['Macro F1']:.4f}")
    print(f"  Distractor Pipeline Precision    : {dist_pipeline_results['Precision']:.4f}")
    print(f"  Distractor Pipeline Recall       : {dist_pipeline_results['Recall']:.4f}")
    print(f"  Distractor Pipeline F1           : {dist_pipeline_results['F1']:.4f}")
    print(f"  Distractor Diversity             : {dist_pipeline_results['Diversity']:.4f}")
    print("\n[SUCCESS] Model B training complete!")


if __name__ == "__main__":
    main()