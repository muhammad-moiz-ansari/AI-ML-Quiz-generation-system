"""
inference.py
============
Unified inference API for Model A and Model B.

This is the ONLY file your Streamlit app needs to import.
It loads all trained models once at startup and exposes
clean functions for the UI to call.

Example:
    result = verify_answer(
        article  = "Pandas eat bamboo...",
        question = "What do pandas eat?",
        option   = "Bamboo"
    )
    print(result)
    # {"correct": True, "confidence": 0.87, "latency_ms": 45, "model_used": "Ensemble"}
"""

import os
import time
import string
import re
import joblib
import numpy as np

# ── Paths - adjust if your folder structure differs ──────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

MODELS_DIR_A    = os.path.join(ROOT_DIR, "models", "model_a", "traditional")
MODELS_DIR_B    = os.path.join(ROOT_DIR, "models", "model_b", "traditional")
ENCODER_PATH    = os.path.join(ROOT_DIR, "models", "onehot_encoder.pkl")

# ── Session log (stored in memory during app runtime) ────────────────────────
# Each inference call appends a record here.
# Screen 4 (Analytics Dashboard) reads from this list.
SESSION_LOG = []


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (runs once when inference.py is first imported)
# ════════════════════════════════════════════════════════════════════════════

def _load_models():
    """
    Load all saved models and the vectorizer into memory.
    Uses a dictionary so we can easily check what's available.

    Returns
    -------
    dict with keys: 'vectorizer', 'ensemble', 'lr', 'svm', 'nb', 'kmeans'
    """
    models = {}

    # Always need the vectorizer - crash loudly if missing
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at {ENCODER_PATH}\n"
            f"Run preprocessing.py first!"
        )
    models['vectorizer'] = joblib.load(ENCODER_PATH)
    print("[inference] [SUCCESS] Vectorizer loaded")

    # Model A - load each, warn if missing (don't crash)
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

    # Model B - load distractor and hint models
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


# Load models at import time - this is the "warm up" step
print("[inference] Loading models...")
MODELS = _load_models()
print("[inference] All available models ready.\n")


# ════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (internal - not called by UI directly)
# ════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    'the','a','an','is','was','are','were','be','been','to','of',
    'in','for','on','with','at','by','from','it','its','this',
    'that','and','or','but','not','no','i','you','we','he','she','they'
}

def _clean(text):
    """Lowercase and remove punctuation - same as preprocessing.py"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def _vectorize(text):
    """
    Convert a raw text string into a One-Hot Encoded feature vector.
    Returns a sparse matrix with shape (1, 5000).
    """
    cleaned = _clean(text)
    return MODELS['vectorizer'].transform([cleaned])


def _log_inference(record: dict):
    """Append an inference record to the session log."""
    record['timestamp'] = time.strftime("%H:%M:%S")
    SESSION_LOG.append(record)


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 1 - verify_answer()
#  Called by: Screen 2 (Quiz View) when user clicks "Check Answer"
# ════════════════════════════════════════════════════════════════════════════

def verify_answer(article: str, question: str, option: str) -> dict:
    """
    Predict whether a chosen answer option is correct.

    Parameters
    ----------
    article  : str  - the full reading passage
    question : str  - the multiple-choice question
    option   : str  - the text of the option the user selected
                      e.g. "Over 99%" not "A"

    Returns
    -------
    dict with keys:
        correct     : bool   - True if model thinks option is correct
        confidence  : float  - probability of being correct (0.0 to 1.0)
        latency_ms  : int    - how long inference took in milliseconds
        model_used  : str    - which model made the prediction
    """

    t_start = time.time()

    # ── Step 1: Build combined feature text ─────────────────────────────
    # Same format as training: article + question + option
    combined_text = (
        _clean(article)  + ' ' +
        _clean(question) + ' ' +
        _clean(option)
    )
    X = MODELS['vectorizer'].transform([combined_text])

    # ── Step 2: Pick best available model ────────────────────────────────
    # Priority: ensemble → svm → lr → nb → fallback
    
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
        # No model available - return a safe fallback
        return {
            "correct"    : False,
            "confidence" : 0.0,
            "latency_ms" : 0,
            "model_used" : "None (models not trained yet)",
            "error"      : "No trained Model A found. Run model_a_train.py first."
        }

    # ── Step 3: Predict ──────────────────────────────────────────────────
    # predict_proba returns [[prob_wrong, prob_correct]]
    proba      = model.predict_proba(X)[0]   # shape: (2,)
    confidence = float(proba[1])             # probability of label=1 (correct)
    predicted  = confidence >= 0.5

    latency_ms = int((time.time() - t_start) * 1000)

    # ── Step 4: Log for analytics dashboard ──────────────────────────────
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
#  Called by: Screen 1 (Article Input) after user clicks "Generate Quiz"
# ════════════════════════════════════════════════════════════════════════════

def generate_question(article: str, gold_question: str = None,
                      gold_options: dict = None, gold_answer: str = None) -> dict:
    """
    Generate a question from an article OR return the RACE gold question.

    If this article came from the RACE dataset (gold_question provided),
    we return the original RACE question + options directly.

    If it's a custom pasted article, we apply template-based generation.

    Parameters
    ----------
    article       : str  - the reading passage
    gold_question : str  - original RACE question (optional)
    gold_options  : dict - {'A': '...', 'B': '...', 'C': '...', 'D': '...'}
    gold_answer   : str  - correct label 'A'/'B'/'C'/'D'

    Returns
    -------
    dict with keys:
        question      : str
        options       : dict  {'A': str, 'B': str, 'C': str, 'D': str}
        correct_label : str   'A'/'B'/'C'/'D'
        correct_text  : str   text of the correct option
        source        : str   'RACE original' or 'Template generated'
    """

    t_start = time.time()

    # ── Case 1: RACE original question available ─────────────────────────
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

    # ── Case 2: Custom article - generate question from scratch ──────────
    generated = _template_generate(article)
    latency_ms = int((time.time() - t_start) * 1000)

    _log_inference({
        "task"       : "generate_question",
        "source"     : "Template generated",
        "latency_ms" : latency_ms,
    })

    return generated

def _template_generate(article: str) -> dict:
    """
    Template-Based Question Generation + Model B Distractor Integration
    """
    import random
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    if not sentences:
        return _fallback_response()

    def score_sentence(sent, idx, total):
        word_count   = len(sent.split())
        length_score = min(word_count / 30, 1.0)        
        pos_score    = 1 - abs((idx / total) - 0.3)     
        return length_score * 0.6 + pos_score * 0.4

    scores    = [score_sentence(s, i, len(sentences)) for i, s in enumerate(sentences)]
    best_idx  = int(np.argmax(scores))
    best_sent = sentences[best_idx]

    words         = [w for w in best_sent.split() if w.lower() not in STOPWORDS]
    answer_phrase = ' '.join(words[:3]) if len(words) >= 3 else ' '.join(words)

    sent_lower = best_sent.lower()
    if any(w in sent_lower for w in ['because', 'reason', 'therefore', 'so that']):
        wh = "Why"
    elif any(w in sent_lower for w in ['when', 'year', 'day', 'time', 'century', 'age']):
        wh = "When"
    elif any(w in sent_lower for w in ['where', 'place', 'city', 'country', 'location']):
        wh = "Where"
    elif any(w in sent_lower for w in ['who', 'person', 'people', 'man', 'woman', 'he', 'she']):
        wh = "Who"
    else:
        wh = "What"

    question = re.sub(
        re.escape(answer_phrase), "_____",
        best_sent, count=1, flags=re.IGNORECASE
    )
    question = f"{wh} {question.strip()}?"
    question = question[:120]   

    # ── INTEGRATE MODEL B DISTRACTORS ──
    distractors = []
    try:
        # Vectorize all sentences and the answer phrase using our One-Hot Encoder
        X_sents = MODELS['vectorizer'].transform(sentences)
        X_ans   = MODELS['vectorizer'].transform([answer_phrase])
        
        # Calculate cosine similarity (How similar is each sentence to the answer?)
        sims = cosine_similarity(X_sents, X_ans).flatten()
        
        # Target "medium" similarity sentences to act as tricky distractors
        scored = [(sentences[i], sims[i]) for i in range(len(sentences))]
        scored.sort(key=lambda x: abs(x[1] - 0.20)) 
        
        for sent, sim in scored:
            # Extract a distractor phrase of similar length
            d_words = [w for w in sent.split() if w.lower() not in STOPWORDS]
            d_phrase = ' '.join(d_words[:3]) if len(d_words) >= 3 else sent
            
            # Make sure we don't accidentally pick the correct answer
            if _clean(d_phrase) != _clean(answer_phrase) and d_phrase not in distractors:
                distractors.append(d_phrase)
                
            if len(distractors) == 3:
                break
    except Exception as e:
        print(f"Distractor generation error: {e}")

    # Fallbacks just in case the article was too short
    fallbacks = ["Not mentioned in the text", "None of the above", "All of the above"]
    while len(distractors) < 3:
        distractors.append(fallbacks[len(distractors)])

    # Shuffle options so the correct answer isn't always 'D'
    all_options = distractors + [answer_phrase]
    random.shuffle(all_options)

    options = {
        "A" : all_options[0],
        "B" : all_options[1],
        "C" : all_options[2],
        "D" : all_options[3],
    }

    # Find which letter holds the correct answer after shuffling
    correct_label = "A"
    for letter, text in options.items():
        if text == answer_phrase:
            correct_label = letter

    return {
        "question"      : question,
        "options"       : options,
        "correct_label" : correct_label,
        "correct_text"  : answer_phrase,
        "source"        : "Template generated + Model B",
    }

def _fallback_response():
    return {
        "question"      : "What is the main idea of the passage?",
        "options"       : {"A": "Option A", "B": "Option B",
                           "C": "Option C", "D": "Option D"},
        "correct_label" : "A",
        "correct_text"  : "Option A",
        "source"        : "Fallback",
    }


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 3 - get_session_metrics()
#  Called by: Screen 4 (Analytics Dashboard)
# ════════════════════════════════════════════════════════════════════════════

def get_session_metrics() -> dict:
    """
    Compute live metrics from the current session's inference log.

    Returns
    -------
    dict with keys:
        total_inferences  : int
        avg_latency_ms    : float
        session_log       : list of dicts (for the table in Screen 4)
        model_a_stats     : dict  (accuracy, avg confidence)
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
        confidences           = [r['confidence'] for r in verify_records]
        model_a_stats['avg_confidence'] = round(float(np.mean(confidences)), 3)
        model_a_stats['total_checks']   = len(verify_records)
        model_a_stats['model_used']     = verify_records[-1].get('model_used', 'N/A')

    return {
        "total_inferences" : len(SESSION_LOG),
        "avg_latency_ms"   : round(float(np.mean(all_latencies)), 1) if all_latencies else 0.0,
        "session_log"      : SESSION_LOG[-20:],   # last 20 records
        "model_a_stats"    : model_a_stats,
    }


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION 4 - get_model_file_metrics()
#  Called by: Screen 4 - loads saved training metrics from CSV
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
#  Called by: Screen 3 (Hint Explorer)
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

    sentences = [s.strip() for s in re.split(r'[.!?]+', article_clean) if len(s.strip().split()) > 3]
    if not sentences:
        return ["Read the passage again.", "Look for the keywords.", "Focus on the main idea."]

    try:
        # Vectorize question
        v_q = MODELS['vectorizer'].transform([question_clean])
        q_words = set(question_clean.split()) - STOPWORDS

        features = []
        for pos, sent in enumerate(sentences):
            # Extract the 4 features Model B was trained on
            v_s      = MODELS['vectorizer'].transform([sent])
            sim_q    = float(cosine_similarity(v_s, v_q))
            s_words  = set(sent.split()) - STOPWORDS
            jacc_q   = len(q_words & s_words) / (len(q_words | s_words) + 1e-9)
            norm_pos = pos / (len(sentences) - 1 + 1e-9)
            norm_len = min(len(sent.split()) / 40.0, 1.0)
            
            features.append([sim_q, jacc_q, norm_pos, norm_len])

        # Predict probability of each sentence being a good hint
        probs = MODELS['hint_scorer'].predict_proba(np.array(features))[:, 1]
        
        # Rank them highest to lowest probability
        ranked = sorted(zip(sentences, probs), key=lambda x: x[1], reverse=True)

        hints = []
        for sent, prob in ranked:
            hints.append(sent)
            if len(hints) == 3:
                break

        # Fallbacks just in case
        fallbacks = ["Read the passage again.", "Focus on the context.", "Look at the actions described."]
        while len(hints) < 3:
            hints.append(fallbacks[len(hints)])

        # Reverse so Hint 1 is the vaguest (lowest prob) and Hint 3 is the best clue
        hints.reverse()
        return hints

    except Exception as e:
        print(f"Hint generation error: {e}")
        return ["Read carefully.", "Check the context.", "Find matching words."]
        

# ════════════════════════════════════════════════════════════════════════════
#  QUICK TEST  (run this file directly to check everything works)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  inference.py - Quick Smoke Test")
    print("="*55)

    SAMPLE_ARTICLE = (
        "The giant panda is a bear native to South Central China. "
        "It is characterised by its bold black-and-white coat. "
        "Though it belongs to the order Carnivora, the giant "
        "panda's diet is over 99% bamboo."
    )
    SAMPLE_QUESTION = "What percentage of the giant panda's diet consists of bamboo?"

    print("\n[TEST 1] verify_answer - correct option")
    r1 = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, "Over 99%")
    print(f"  Result : {r1}")

    print("\n[TEST 2] verify_answer - wrong option")
    r2 = verify_answer(SAMPLE_ARTICLE, SAMPLE_QUESTION, "About 75%")
    print(f"  Result : {r2}")

    print("\n[TEST 3] generate_question - custom article")
    r3 = generate_question(SAMPLE_ARTICLE)
    print(f"  Question : {r3['question']}")
    print(f"  Answer   : {r3['correct_text']}")
    print(f"  Source   : {r3['source']}")

    print("\n[TEST 4] get_session_metrics")
    metrics = get_session_metrics()
    print(f"  Total inferences : {metrics['total_inferences']}")
    print(f"  Avg latency      : {metrics['avg_latency_ms']}ms")

    print("\n[SUCCESS] Smoke test complete!")
