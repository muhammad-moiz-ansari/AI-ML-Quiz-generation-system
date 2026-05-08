"""
inference.py
============
Unified inference API for Model A and Model B.
"""

import os
import time
import string
import re
import random
import joblib
import numpy as np
import scipy.sparse

# ── Paths - adjust if your folder structure differs ──────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

MODELS_DIR_A    = os.path.join(ROOT_DIR, "models", "model_a", "traditional")
MODELS_DIR_B    = os.path.join(ROOT_DIR, "models", "model_b", "traditional")
ENCODER_PATH    = os.path.join(ROOT_DIR, "models", "onehot_encoder.pkl")

# ── Session log (stored in memory during app runtime) ────────────────────────
SESSION_LOG = []


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (runs once when inference.py is first imported)
# ════════════════════════════════════════════════════════════════════════════

def _load_models():
    """
    Load all saved models and the vectorizer into memory.
    Using a dictionary so we can easily check what's available.
    """
    models = {}

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at {ENCODER_PATH}\n"
            f"Run preprocessing.py first!"
        )
    models['vectorizer'] = joblib.load(ENCODER_PATH)
    print("[inference] [SUCCESS] Vectorizer loaded")

    model_a_files = {
        'ensemble' : 'ensemble_model.pkl',
        'lr'       : 'lr_model.pkl',
        'svm'      : 'svm_model.pkl',
        'nb'       : 'nb_model.pkl',
        'kmeans'   : 'kmeans_model.pkl',
    }

    for key, filename in model_a_files.items():
        path = os.path.join(MODELS_DIR_A, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
            print(f"[inference] [SUCCESS] Model A - {key} loaded")
        else:
            models[key] = None
            print(f"[inference] [WARNING] Model A - {key} NOT FOUND (run model_a_train.py)")

    model_b_files = {
        'distractor_ranker' : 'distractor_lr_ranker.pkl',
        'hint_scorer'       : 'hint_scorer.pkl',
    }

    for key, filename in model_b_files.items():
        path = os.path.join(MODELS_DIR_B, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
            print(f"[inference] [SUCCESS] Model B - {key} loaded")
        else:
            models[key] = None
            print(f"[inference] [WARNING] Model B - {key} NOT FOUND (run model_b_train.py)")

    return models


print("[inference] Loading models...")
MODELS = _load_models()
print("[inference] All available models ready.\n")

# ── Detect how many features the saved models expect ─────────────────────────
# (trained with _hand_crafted_features): 5007
def _detect_n_features():
    for key in ('lr', 'svm', 'nb', 'ensemble'):
        m = MODELS.get(key)
        if m is None:
            continue
        # VotingClassifier wraps estimators; inspect the first named step
        try:
            if hasattr(m, 'n_features_in_'):
                return int(m.n_features_in_)
            # CalibratedClassifierCV
            if hasattr(m, 'estimators_'):
                inner = m.estimators_[0]
                if hasattr(inner, 'coef_'):
                    return int(inner.coef_.shape[1])
            # VotingClassifier
            if hasattr(m, 'estimators'):
                for _, sub in m.estimators:
                    if hasattr(sub, 'n_features_in_'):
                        return int(sub.n_features_in_)
        except Exception:
            pass
    return 5007   # default to new if unknown

N_FEATURES_EXPECTED = _detect_n_features()
print(f"[inference] Models expect {N_FEATURES_EXPECTED} features.")


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    'the','a','an','is','was','are','were','be','been','to','of',
    'in','for','on','with','at','by','from','it','its','this',
    'that','and','or','but','not','no','i','you','we','he','she','they',
    'has','have','had','do','did','will','would','could','should',
    'as','if','than','then','so','up','out','about','into','also'
}

def _clean(text):
    """Lowercase and remove punctuation - same as preprocessing.py"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


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
    Features (7 dimensions):
      0  keyword overlap between option and article
      1  keyword overlap between option and question
      2  keyword overlap between question and article
      3  length of option normalised by max option length in the group
      4  position of this option in [A,B,C,D] list  (0.0 to 1.0)
      5  binary: option text appears verbatim as substring of article
      6  fraction of option unique words present in article
    """
    art_clean  = article.lower().translate(str.maketrans('', '', string.punctuation))
    q_clean    = question.lower().translate(str.maketrans('', '', string.punctuation))
    opt_clean  = option.lower().translate(str.maketrans('', '', string.punctuation))

    f0 = _keyword_overlap(opt_clean, art_clean)
    f1 = _keyword_overlap(opt_clean, q_clean)
    f2 = _keyword_overlap(q_clean, art_clean)

    max_len = max(len(o.split()) for o in all_options) if all_options else 1
    f3 = len(option.split()) / (max_len + 1e-9)

    try:
        pos = all_options.index(option)
        f4  = pos / max(len(all_options) - 1, 1)
    except ValueError:
        f4 = 0.0

    f5 = 1.0 if opt_clean in art_clean else 0.0

    opt_words = set(opt_clean.split()) - STOPWORDS
    art_words = set(art_clean.split()) - STOPWORDS
    f6 = len(opt_words & art_words) / (len(opt_words) + 1e-9) if opt_words else 0.0

    return np.array([f0, f1, f2, f3, f4, f5, f6], dtype=np.float32)


def _build_feature_vector(article: str, question: str, option: str,
                           all_options: list = None):
    """
    Build the full 5007-feature vector used by the trained models:
      - 5000 OHE features  (article + question + option combined text)
      - 7 hand-crafted relationship features
    """
    combined_text = _clean(article) + ' ' + _clean(question) + ' ' + _clean(option)
    X_ohe = MODELS['vectorizer'].transform([combined_text])   # (1, 5000) sparse

    if all_options is None:
        all_options = [option]

    hc   = _hand_crafted_features(article, question, option, all_options)
    X_hc = scipy.sparse.csr_matrix(hc.reshape(1, -1))        # (1, 7) sparse

    if N_FEATURES_EXPECTED == 5000:
        # Old models: return OHE only to stay compatible
        return X_ohe
    else:
        # New models: stack OHE + hand-crafted
        return scipy.sparse.hstack([X_ohe, X_hc], format='csr')


def _log_inference(record: dict):
    """Append an inference record to the session log."""
    record['timestamp'] = time.strftime("%H:%M:%S")
    SESSION_LOG.append(record)


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 1 - verify_answer()
# ════════════════════════════════════════════════════════════════════════════

def verify_answer(article: str, question: str, option: str) -> dict:
    """
    Predict whether a chosen answer option is correct.

    Parameters
    ----------
    article  : str  - the full reading passage
    question : str  - the multiple-choice question
    option   : str  - the text of the option the user selected

    Returns
    -------
    dict with keys: correct, confidence, latency_ms, model_used
    """

    t_start = time.time()

    # Build the same feature vector used during training.
    # all_options=None is fine for verification - the position/length
    # features will be computed relative to this single option only,
    # which is consistent because the model sees one option at a time.
    X = _build_feature_vector(article, question, option, all_options=[option])

    if MODELS.get('lr') is not None:
        model      = MODELS['lr']
        model_name = "Logistic Regression"
    elif MODELS.get('ensemble') is not None:
        model      = MODELS['ensemble']
        model_name = "Soft-Vote Ensemble"
    elif MODELS.get('svm') is not None:
        model      = MODELS['svm']
        model_name = "SVM"
    elif MODELS.get('nb') is not None:
        model      = MODELS['nb']
        model_name = "Naive Bayes"
    else:
        return {
            "correct"    : False,
            "confidence" : 0.0,
            "latency_ms" : 0,
            "model_used" : "None (models not trained yet)",
            "error"      : "No trained Model A found. Run model_a_train.py first."
        }

    proba      = model.predict_proba(X)[0]   # shape: (2,)
    confidence = float(proba[1])
    predicted  = confidence >= 0.5

    latency_ms = int((time.time() - t_start) * 1000)

    _log_inference({
        "task"       : "verify_answer",
        "question"   : question[:60] + "..." if len(question) > 60 else question,
        "option"     : option,
        "predicted"  : "Correct" if predicted else "Wrong",
        "confidence" : round(confidence, 3),
        "latency_ms" : latency_ms,
        "model_used" : model_name,
    })

    return {
        "correct"    : bool(predicted),
        "confidence" : round(confidence, 3),
        "latency_ms" : latency_ms,
        "model_used" : model_name,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 2 - generate_question()
# ════════════════════════════════════════════════════════════════════════════

def generate_question(article: str, gold_question: str = None,
                      gold_options: dict = None, gold_answer: str = None) -> dict:
    """
    Generate a question from an article OR return the RACE gold question.
    """

    t_start = time.time()

    if gold_question and gold_options and gold_answer:
        latency_ms = int((time.time() - t_start) * 1000)
        _log_inference({
            "task"       : "generate_question",
            "source"     : "RACE original",
            "latency_ms" : latency_ms,
        })
        return {
            "question"      : gold_question,
            "options"       : gold_options,
            "correct_label" : gold_answer,
            "correct_text"  : gold_options[gold_answer],
            "source"        : "RACE original",
        }

    generated  = _template_generate(article)
    latency_ms = int((time.time() - t_start) * 1000)

    _log_inference({
        "task"       : "generate_question",
        "source"     : generated.get("source", "Template generated"),
        "latency_ms" : latency_ms,
    })

    return generated


# ════════════════════════════════════════════════════════════════════════════
#  CORE: TEMPLATE-BASED QUESTION GENERATION
#
#  Step 1 - Extract candidate sentences using One-Hot keyword overlap with the correct answer phrase.
#  Step 2 - Apply Wh-word templates to transform each sentence into a question.
#  Step 3 - Rank all generated questions using the trained SVM (or LR) as an ML scoring model and
#           pick the highest-scoring one.
# ════════════════════════════════════════════════════════════════════════════

def _pick_answer_phrase(sentence: str) -> str:
    """
    Choose the ANSWER PHRASE from a sentence.

    Strategy: Find the longest noun-phrase-like chunk (consecutive non-stopword tokens) in the sentence.  
    This tends to be a meaningful entity or fact rather than a stopword-heavy fragment.

    Returns the raw (un-cleaned) answer text so it matches the original sentence for the 
    blank-substitution step.
    """
    tokens = sentence.split()
    best_chunk = []
    current_chunk = []

    for tok in tokens:
        tok_clean = re.sub(r'[^\w]', '', tok).lower()
        if tok_clean and tok_clean not in STOPWORDS and len(tok_clean) > 1:
            current_chunk.append(tok)
        else:
            if len(current_chunk) > len(best_chunk):
                best_chunk = current_chunk[:]
            current_chunk = []
    if len(current_chunk) > len(best_chunk):
        best_chunk = current_chunk[:]

    if not best_chunk:
        return ""

    # Keep the phrase to at most 5 words so it is a tight, quiz-able answer
    return ' '.join(best_chunk[:5])


def _apply_wh_template(sentence: str, answer_phrase: str) -> str:
    """
    Substitute the answer phrase with _____ and prepend the correct Wh-word.

    Returns a well-formed question string, or "" if substitution fails.
    """
    sent_lower = sentence.lower()

    # Choose Wh-word based on contextual clues in the sentence
    if any(kw in sent_lower for kw in ['because', 'reason', 'therefore', 'so that', 'in order']):
        wh = "Why"
    elif any(kw in sent_lower for kw in ['when', 'year', 'century', 'decade', 'day', 'month',
                                          'morning', 'evening', 'night', 'age', 'period']):
        wh = "When"
    elif any(kw in sent_lower for kw in ['where', 'place', 'city', 'town', 'country',
                                          'location', 'region', 'area', 'street']):
        wh = "Where"
    elif any(kw in sent_lower for kw in ['who', 'person', 'people', 'man', 'woman',
                                          'boy', 'girl', 'teacher', 'student', 'father',
                                          'mother', 'friend', 'author', 'writer']):
        wh = "Who"
    elif any(kw in sent_lower for kw in ['how many', 'how much', 'number', 'amount',
                                          'percent', '%', 'total', 'count']):
        wh = "How many"
    else:
        wh = "What"

    # Substitute the answer phrase with a blank (case-insensitive)
    pattern = re.compile(re.escape(answer_phrase), re.IGNORECASE)
    q_body  = pattern.sub("_____", sentence, count=1)

    if "_____" not in q_body:
        # The phrase was not found verbatim - skip this candidate
        return ""

    # Strip leading articles/lowercase words that look odd after Wh-word
    q_body = q_body.strip()
    q_body = re.sub(r'^(the|a|an)\s+', '', q_body, flags=re.IGNORECASE)

    # Capitalise the first letter after the Wh-word
    q_body = q_body[0].upper() + q_body[1:] if q_body else q_body

    question = f"{wh} {q_body}"

    # Ensure it ends with exactly one question mark
    question = question.rstrip('.!?') + '?'

    return question


def _template_generate(article: str) -> dict:
    """
    Full question-generation pipeline

    Step 1 - Score every sentence by its One-Hot keyword overlap with its own
             answer phrase (used as the candidate-selection criterion).
    Step 2 - Apply Wh-word templates to produce a candidate question per
             sentence.
    Step 3 - Rank all valid candidates with the trained SVM/LR ML ranker
             (predict_proba score) and pick the best one.
    Then   - Build 3 distractors using cosine similarity on OHE vectors
             (sentences at medium similarity to the answer phrase).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # ── Split article into clean sentences ──────────────────────────────────
    raw_sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    sentences     = [s.strip() for s in raw_sentences if len(s.split()) >= 5]

    if not sentences:
        return _fallback_response()

    # ── ML Ranker: prefer SVM (as required by rubric), fall back to LR ──────
    ranker_model = MODELS.get('svm') or MODELS.get('lr')

    # ── STEP 1 + 2 + 3: Build & rank all candidate questions ────────────────
    candidate_questions = []

    for sent in sentences:
        answer_phrase = _pick_answer_phrase(sent)
        if not answer_phrase or len(answer_phrase.split()) < 1:
            continue

        # --- STEP 1: Compute One-Hot keyword overlap between sentence & answer ---
        try:
            v_sent  = MODELS['vectorizer'].transform([_clean(sent)])
            v_ans   = MODELS['vectorizer'].transform([_clean(answer_phrase)])
            if v_sent.nnz == 0 or v_ans.nnz == 0:
                ohe_overlap = 0.0
            else:
                ohe_overlap = float(cosine_similarity(v_sent, v_ans)[0][0])
        except Exception:
            ohe_overlap = 0.0

        # Skip sentences where the answer phrase has zero overlap with OHE vocab
        # (the phrase is completely out-of-vocabulary - won't produce a good question)
        if ohe_overlap == 0.0:
            continue

        # --- STEP 2: Apply Wh-word template ---
        question_text = _apply_wh_template(sent, answer_phrase)
        if not question_text:
            # Template substitution failed for this sentence - skip
            continue

        # Truncate to a readable length
        if len(question_text) > 130:
            question_text = question_text[:127] + "...?"

        # --- STEP 3: Rank with ML classifier ---
        try:
            X_rank = _build_feature_vector(article, question_text, answer_phrase,
                                           all_options=[answer_phrase])

            if ranker_model is not None:
                ml_score = float(ranker_model.predict_proba(X_rank)[0][1])
            else:
                ml_score = ohe_overlap

        except Exception as e:
            print(f"[inference] [WARNING] ML ranker error for sentence: {e}")
            ml_score = ohe_overlap

        candidate_questions.append({
            'question'     : question_text,
            'answer'       : answer_phrase,
            'ml_score'     : ml_score,
            'ohe_overlap'  : ohe_overlap,
            'sentence'     : sent,
        })

    if not candidate_questions:
        return _fallback_response()

    # ── Pick the best candidate according to the ML ranker score ────────────
    candidate_questions.sort(key=lambda x: x['ml_score'], reverse=True)
    best = candidate_questions[0]

    question      = best['question']
    answer_phrase = best['answer']

    print(f"[inference] [SUCCESS] Best question selected (ML score={best['ml_score']:.4f}): {question}")
    print(f"[inference]           Answer phrase: {answer_phrase}")

    # ════════════════════════════════════════════════════════════════════════
    #  DISTRACTOR GENERATION (Model B pipeline - OHE cosine similarity)
    #  Find sentences that are MODERATELY similar to the answer (plausible
    #  but not correct).  Extract a short, clean answer-length phrase from each.
    # ════════════════════════════════════════════════════════════════════════

    distractors = _generate_distractors(sentences, answer_phrase)

    # ── Build options dict and shuffle so correct answer is not always 'D' ──
    all_options = distractors + [answer_phrase]
    random.shuffle(all_options)

    options = {
        "A": all_options[0],
        "B": all_options[1],
        "C": all_options[2],
        "D": all_options[3],
    }

    correct_label = next(
        (letter for letter, text in options.items() if text == answer_phrase),
        "A"
    )

    return {
        "question"      : question,
        "options"       : options,
        "correct_label" : correct_label,
        "correct_text"  : answer_phrase,
        "source"        : "ML Ranked Template + Model B Distractors",
    }


def _generate_distractors(sentences: list, answer_phrase: str) -> list:
    """
    Generate exactly 3 distractors using One-Hot Encoded cosine similarity.

    Selects sentences whose similarity to the answer phrase is in the
    'medium' range (~0.15-0.40): plausible but not the actual answer.
    Extracts a clean, answer-length phrase from each distractor sentence
    so the options look uniform on the quiz screen.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    distractors     = []
    fallbacks       = [
        "Not mentioned in the passage",
        "The opposite of what was described",
        "None of the above",
    ]

    try:
        v_ans    = MODELS['vectorizer'].transform([_clean(answer_phrase)])
        v_sents  = MODELS['vectorizer'].transform([_clean(s) for s in sentences])
        sims     = cosine_similarity(v_sents, v_ans).flatten()

        # Sort by closeness to the target similarity of 0.25
        # (not too close → would give away answer; not too far → obviously wrong)
        scored = sorted(
            zip(sentences, sims),
            key=lambda x: abs(x[1] - 0.25)
        )

        ans_clean  = _clean(answer_phrase)
        ans_words  = set(ans_clean.split()) - STOPWORDS
        ans_length = len(answer_phrase.split())   # aim for similar phrase length

        for sent, sim in scored:
            # Never use the sentence that CONTAINS the answer (too revealing)
            if ans_clean in _clean(sent):
                continue

            # Extract a distractor phrase of similar length to the answer
            d_phrase = _extract_phrase_from_sentence(sent, ans_length)

            if not d_phrase:
                continue

            # Ensure it is not semantically identical to the correct answer
            d_clean = _clean(d_phrase)
            d_words = set(d_clean.split()) - STOPWORDS
            if not d_words:
                continue

            overlap = (
                len(ans_words & d_words) / (len(ans_words | d_words) + 1e-9)
                if ans_words or d_words else 0.0
            )
            if overlap > 0.7:
                continue

            # Ensure diversity among already-selected distractors
            too_similar = False
            for prev in distractors:
                prev_words = set(_clean(prev).split()) - STOPWORDS
                jaccard    = (
                    len(d_words & prev_words) / (len(d_words | prev_words) + 1e-9)
                    if d_words or prev_words else 0.0
                )
                if jaccard > 0.5:
                    too_similar = True
                    break

            if too_similar:
                continue

            if d_phrase not in distractors:
                distractors.append(d_phrase)

            if len(distractors) == 3:
                break

    except Exception as e:
        print(f"[inference] [WARNING] Distractor generation error: {e}")

    # Pad with readable fallbacks if fewer than 3 were found
    while len(distractors) < 3:
        distractors.append(fallbacks[len(distractors)])

    return distractors


def _extract_phrase_from_sentence(sentence: str, target_length: int = 3) -> str:
    """
    Extract a meaningful noun-phrase-like chunk from a sentence that is
    approximately target_length words long.
    """
    tokens = sentence.split()
    chunks = []
    current_chunk = []

    for tok in tokens:
        tok_clean = re.sub(r'[^\w]', '', tok).lower()
        if tok_clean and tok_clean not in STOPWORDS and len(tok_clean) > 1:
            current_chunk.append(tok)
        else:
            if current_chunk:
                chunks.append(current_chunk[:])
            current_chunk = []
    if current_chunk:
        chunks.append(current_chunk[:])

    if not chunks:
        return ""

    # Pick the chunk whose length is closest to target_length
    best_chunk = min(chunks, key=lambda c: abs(len(c) - target_length))
    return ' '.join(best_chunk[:max(target_length, 2)])


def _fallback_response() -> dict:
    return {
        "question"      : "What is the main idea of the passage?",
        "options"       : {
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
        },
        "correct_label" : "A",
        "correct_text"  : "Option A",
        "source"        : "Fallback",
    }


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 3 - get_session_metrics()
# ════════════════════════════════════════════════════════════════════════════

def get_session_metrics() -> dict:
    """
    Compute live metrics from the current session's inference log.
    """
    if not SESSION_LOG:
        return {
            "total_inferences" : 0,
            "avg_latency_ms"   : 0.0,
            "session_log"      : [],
            "model_a_stats"    : {"note": "No inferences yet"},
        }

    verify_records = [r for r in SESSION_LOG if r.get('task') == 'verify_answer']
    all_latencies  = [r['latency_ms'] for r in SESSION_LOG if 'latency_ms' in r]

    model_a_stats = {}
    if verify_records:
        confidences                     = [r['confidence'] for r in verify_records]
        model_a_stats['avg_confidence'] = round(float(np.mean(confidences)), 3)
        model_a_stats['total_checks']   = len(verify_records)
        model_a_stats['model_used']     = verify_records[-1].get('model_used', 'N/A')

    return {
        "total_inferences" : len(SESSION_LOG),
        "avg_latency_ms"   : round(float(np.mean(all_latencies)), 1) if all_latencies else 0.0,
        "session_log"      : SESSION_LOG[-20:],
        "model_a_stats"    : model_a_stats,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 4 - get_model_file_metrics()
# ════════════════════════════════════════════════════════════════════════════

def get_model_file_metrics() -> dict:
    """
    Load the saved model comparison table from training.
    Returns the results DataFrame as a list of dicts.
    """
    results_path = os.path.join(MODELS_DIR_A, 'model_a_results.csv')

    if not os.path.exists(results_path):
        return {"error": "model_a_results.csv not found. Run model_a_train.py first."}

    import pandas as pd
    df = pd.read_csv(results_path)
    return {"model_a_results": df.to_dict(orient='records')}


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 5 - get_hints()
# ════════════════════════════════════════════════════════════════════════════

def get_hints(article: str, question: str) -> list:
    """
    Generate 3 graduated hints using Model B's trained Hint Scorer.
    Hint 1 = vaguest, Hint 3 = most explicit.
    """
    if MODELS.get('hint_scorer') is None:
        return ["Hint model not loaded."] * 3

    from sklearn.metrics.pairwise import cosine_similarity

    article_clean  = _clean(article)
    question_clean = _clean(question)

    sentences = [s.strip() for s in re.split(r'[.!?]+', article_clean)
                 if len(s.strip().split()) > 3]
    if not sentences:
        return ["Read the passage again.", "Look for the keywords.", "Focus on the main idea."]

    try:
        v_q    = MODELS['vectorizer'].transform([question_clean])
        q_words = set(question_clean.split()) - STOPWORDS

        # Detect how many features the hint scorer expects (4 old or 6 new)
        hint_n_features = getattr(MODELS['hint_scorer'], 'n_features_in_', 6)

        ans_clean = _clean(question)   # approximate: use question as ans proxy if answer unknown
        ans_words = set(ans_clean.split()) - STOPWORDS

        features = []
        for pos, sent in enumerate(sentences):
            v_s      = MODELS['vectorizer'].transform([sent])
            sim_q    = float(cosine_similarity(v_s, v_q)[0][0])
            s_words  = set(sent.split()) - STOPWORDS
            jacc_q   = len(q_words & s_words) / (len(q_words | s_words) + 1e-9)
            norm_pos = pos / (len(sentences) - 1 + 1e-9)
            norm_len = min(len(sent.split()) / 40.0, 1.0)

            if hint_n_features >= 6:
                jacc_ans   = len(ans_words & s_words) / (len(ans_words | s_words) + 1e-9) if ans_words else 0.0
                has_ans_kw = 1.0 if (ans_words and ans_words & s_words) else 0.0
                features.append([sim_q, jacc_q, norm_pos, norm_len, jacc_ans, has_ans_kw])
            else:
                features.append([sim_q, jacc_q, norm_pos, norm_len])

        probs  = MODELS['hint_scorer'].predict_proba(np.array(features))[:, 1]
        ranked = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)

        hints = []
        for sent, prob in ranked:
            hints.append(sent)
            if len(hints) == 3:
                break

        fallbacks = ["Read the passage again.", "Focus on the context.", "Look at the actions described."]
        while len(hints) < 3:
            hints.append(fallbacks[len(hints)])

        hints.reverse()
        return hints

    except Exception as e:
        print(f"[inference] [WARNING] Hint generation error: {e}")
        return ["Read carefully.", "Check the context.", "Find matching words."]


# ════════════════════════════════════════════════════════════════════════════
#  QUICK TEST  (run this file directly to verify everything works)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  inference.py - Quick Smoke Test")
    print("="*55)

    SAMPLE_ARTICLE = (
        "The giant panda is a bear native to South Central China. "
        "It is characterised by its bold black-and-white coat and large body. "
        "Though it belongs to the order Carnivora, the giant panda's diet "
        "is over 99 percent bamboo. Pandas spend 10 to 16 hours each day "
        "eating bamboo shoots, leaves, and stems. "
        "The giant panda lives in a few mountain ranges in central China, "
        "mainly in Sichuan province. It is a national symbol of China and "
        "is used in the logo of the World Wildlife Fund."
    )
    SAMPLE_QUESTION = "What percentage of the giant panda's diet consists of bamboo?"

    print("\n[TEST 1] verify_answer - correct option")
    r1 = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, "Over 99 percent")
    print(f"  Result : {r1}")

    print("\n[TEST 2] verify_answer - wrong option")
    r2 = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, "About 75 percent")
    print(f"  Result : {r2}")

    print("\n[TEST 3] generate_question - custom article")
    r3 = generate_question(SAMPLE_ARTICLE)
    print(f"  Question      : {r3['question']}")
    print(f"  Correct Answer: {r3['correct_text']}")
    print(f"  Options       : {r3['options']}")
    print(f"  Source        : {r3['source']}")

    print("\n[TEST 4] get_session_metrics")
    metrics = get_session_metrics()
    print(f"  Total inferences : {metrics['total_inferences']}")
    print(f"  Avg latency      : {metrics['avg_latency_ms']}ms")

    print("\n[SUCCESS] Smoke test complete!")