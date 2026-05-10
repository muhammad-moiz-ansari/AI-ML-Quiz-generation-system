# ⚡ AI-ML Quiz Generation System

> **Intelligent Reading Comprehension and Quiz Generation System**  
> Built on the RACE dataset using classical Machine Learning and One-Hot Encoding.  
> No deep learning. No transformers. Pure scikit-learn.

---

## 👥 Group Members

| Name | Roll Number |
|------|-------------|
| Moiz Ansari | 23i-0523 |
| M. Ali Sher | 23i-0683 |

**Course:** AL2002 — Artificial Intelligence Lab  
**Campus:** FAST NUCES, Islamabad  
**Year:** 2026

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Details](#model-details)
- [Training Results](#training-results)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Training Instructions](#training-instructions)
- [Running the App](#running-the-app)
- [UI Screens](#ui-screens)
- [Evaluation Metrics](#evaluation-metrics)
- [Known Limitations](#known-limitations)

---

## 🧠 Project Overview

This system automatically reads an English passage, generates a multiple-choice comprehension question, produces three plausible-but-wrong distractor options, verifies whether a user's selected answer is correct, and provides graduated hints when the user is wrong.

The entire pipeline uses **classical ML only** (scikit-learn), with **One-Hot Encoding (OHE)** as the primary feature representation, as required by the project rubric. No BERT, no transformers, no neural networks.

**Dataset:** [RACE](https://www.cs.cmu.edu/~glai1/data/race/) — Reading Comprehension Dataset from Examinations. 87,866 training questions drawn from Chinese English-language exams, each paired with a passage and four answer options (one correct, three distractors).

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                │
│  Screen 1: Article Input   Screen 2: Take Quiz          │
│  Screen 3: Hint Explorer   Screen 4: Analytics          │
└───────────────────────┬─────────────────────────────────┘
                        │ calls
┌───────────────────────▼─────────────────────────────────┐
│                   inference.py (Unified API)            │
│  verify_answer()   generate_question()   get_hints()    │
└──────────┬────────────────────────────────┬─────────────┘
           │                                │
┌──────────▼───────────┐        ┌───────────▼──────────────┐
│      MODEL A         │        │       MODEL B            │
│  Answer Verifier     │        │  Distractor Generator    │
│  + Question Gen      │        │  + Hint Scorer           │
│                      │        │                          │
│  • Logistic Regr.    │        │  • LR Distractor Ranker  │
│  • SVM (LinearSVC)   │        │  • RF Distractor Ranker  │
│  • Bernoulli NB      │        │  • LR Hint Scorer        │
│  • Soft-Vote Ensemble│        │  • OHE Cosine Similarity │
│  • K-Means           │        │    Pipeline              │
│  • Label Propagation │        └──────────────────────────┘
└──────────────────────┘
           │
┌──────────▼───────────────────────────────────────────────┐
│              onehot_encoder.pkl  (shared)                │
│   CountVectorizer(binary=True, max_features=5000)        │
│   Vocabulary built from RACE training articles+questions │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis (EDA)

Before modeling, a comprehensive exploratory analysis was conducted on the RACE dataset to understand its structural properties. The full analysis, visualizations, and summary statistics can be found in `notebooks/EDA.ipynb`.

Key insights derived from the EDA include:
- **Passage & Question Lengths:** Analysis of word count distributions to inform the `max_features` limit for our One-Hot Encoder.
- **Answer Distribution:** Verification of the 1:3 class imbalance (25% correct vs 75% wrong options).
- **Question Types:** Extraction of Wh-word distributions (What, Why, How, etc.) which informed the design of our template-based Question Generator.
- **Lexical Overlap:** Analysis of Jaccard similarity between correct answers and passages, which justified our feature engineering choices for Model B's Hint Scorer.

---

## 🔬 Model Details

### Preprocessing (`src/preprocessing.py`)

- Lowercases all text and removes punctuation
- Cleans six columns per RACE row: `article`, `question`, `A`, `B`, `C`, `D`
- Original columns preserved alongside `clean_*` columns for UI display
- Fits a single shared `CountVectorizer(binary=True, max_features=5000)` on training articles + questions
- Saves `onehot_encoder.pkl` to `models/` for use by both Model A and Model B

### Model A — Answer Verifier + Question Generator (`src/model_a_train.py`)

**Answer Verifier (binary classification):**

Each RACE row is expanded into 4 binary classification examples — one per answer option. Label = 1 if the option is the correct answer, 0 otherwise. This gives a 75/25 class imbalance (3 wrong options per 1 correct) which is handled via `class_weight='balanced'` and `sample_weight`.

Feature vector per example = **5007 dimensions**:
- 5000 OHE binary features (article + question + option concatenated)
- 7 hand-crafted relationship features:
  - Keyword overlap (Jaccard) between option and article
  - Keyword overlap between option and question
  - Keyword overlap between question and article
  - Normalised option length relative to the 4 options
  - Option position (A=0.0, B=0.33, C=0.67, D=1.0)
  - Binary: does the option appear verbatim in the article?
  - Fraction of option keywords present in the article

Models trained:
| Model | Notes |
|-------|-------|
| Logistic Regression | C=1.0, class_weight='balanced', solver='lbfgs' |
| SVM (LinearSVC) | C=1.0, class_weight='balanced', wrapped in CalibratedClassifierCV for probabilities |
| Bernoulli Naive Bayes | alpha=1.0, trained with compute_sample_weight('balanced') |
| K-Means | Unsupervised, k=2, evaluated with Silhouette Score |
| Label Propagation | Semi-supervised, knn kernel, 80% labels masked |
| Soft-Vote Ensemble | LR×2 + SVM×1 + NB×1 weighted voting |

**Question Generator (template-based + ML ranking):**

Implements the 3-step rubric pipeline:
1. Extract candidate sentences from the passage and score each by OHE cosine similarity between the sentence and its candidate answer phrase
2. Apply Wh-word templates (Who/What/Where/When/Why/How many) by substituting the answer phrase with `_____` and prepending the appropriate question word
3. Rank all generated candidate questions using the trained SVM classifier (`predict_proba` score) and select the highest-scoring one

### Model B — Distractor Generator + Hint Scorer (`src/model_b_train.py`)

**Distractor Generator:**

- Extracts phrase-length chunks from the article (matching answer length, 50% overlap) as candidates
- Scores each candidate by OHE cosine similarity to the correct answer
- Selects 3 candidates with similarity closest to 0.25 (plausible but not correct)
- Applies diversity filtering (Jaccard distance > 0.15 between selected distractors)

**LR/RF Distractor Ranker (6 features):**
- Cosine similarity of candidate to correct answer
- Cosine similarity of candidate to question
- Jaccard keyword overlap with correct answer
- Jaccard keyword overlap with question
- Normalised candidate length
- Binary: candidate shares any keyword with correct answer

**Hint Scorer (6 features, LR):**
- Cosine similarity of sentence to question
- Jaccard keyword overlap with question
- Sentence position in article (normalised)
- Sentence length (normalised)
- Jaccard keyword overlap with correct answer ← key feature
- Binary: sentence contains any answer keyword ← key feature

Label: a sentence is a "good hint" if it contains >40% of the correct answer's keywords.

---

## 📊 Training Results

### Model A — Answer Verifier (Dev Set, 351,464 examples)

| Model | Accuracy | Macro F1 | F1 (Correct class) |
|-------|----------|----------|-------------------|
| Logistic Regression | 56.23% | 0.5225 | **0.3847** |
| SVM (LinearSVC) | 75.00% | 0.4286 | 0.0001 |
| Bernoulli Naive Bayes | 47.75% | 0.4656 | **0.3860** |
| Weighted Soft-Vote Ensemble | 74.65% | 0.4478 | 0.0417 |
| K-Means Silhouette Score | — | — | 0.0128 |
| Label Propagation Accuracy | 72.31% | 0.4196 | — |

> **Note:** LR and NB are the only models genuinely learning both classes. SVM collapses to the majority class (75% = always predict "Wrong") due to the 3:1 class imbalance inherent in 4-choice MCQs. The F1 on the "Correct" class is the primary metric because the dataset has 75% wrong options by construction.

### Model A — Question Generator (NLP Metrics, 200 Dev Samples)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| BLEU-2 | 0.0501 | Within expected range for template-based generation (0.05–0.15) |
| ROUGE-L | 0.1514 | Measures longest common subsequence with gold questions |
| METEOR | 0.1516 | Accounts for stemming and synonyms |

> Per the instructor's note: since question generation is a text generation task, BLEU/ROUGE/METEOR are the appropriate evaluation metrics rather than classification accuracy.

### Model B — Distractor Ranker (527,180 examples, 6 features)

| Model | Accuracy | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| LR Ranker | 62.28% | 0.5965 | 0.7512 | 0.3673 |
| RF Ranker | 77.80% | 0.7727 | 0.9011 | 0.6246 |

### Model B — Hint Scorer (87,861 examples, 6 features)

| Metric | Score |
|--------|-------|
| Accuracy | 86.57% |
| Macro F1 | 0.8355 |
| Precision | 0.8676 |
| Recall | 0.9480 |

### Model B — Distractor Pipeline (500 Dev Samples, Partial Keyword Jaccard ≥ 0.30)

| Metric | Score |
|--------|-------|
| Precision | 0.0473 |
| Recall | 0.0490 |
| F1 | 0.0482 |
| Accuracy (no answer leak) | 1.0000 |
| Diversity (Jaccard distance) | 0.9737 |

> Pipeline precision/recall is low but expected: template-generated phrase distractors rarely share ≥30% keywords with gold RACE distractor options. Diversity of 0.97 confirms the 3 generated distractors are highly distinct from each other.

---

## 📁 Project Structure

```
AI-ML-Quiz-generation-system/
│
├── data/
│   ├── raw/                    # Original RACE CSVs (not committed to git)
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   └── test.csv
│   └── processed/              # Generated by preprocessing.py
│       ├── clean_train.csv
│       ├── clean_dev.csv
│       ├── clean_test.csv
│       └── train_features.npz
│
├── models/
│   ├── onehot_encoder.pkl      # Shared OHE vectorizer (5000 features)
│   ├── model_a/
│   │   └── traditional/
│   │       ├── lr_model.pkl
│   │       ├── svm_model.pkl
│   │       ├── nb_model.pkl
│   │       ├── kmeans_model.pkl
│   │       ├── lp_model.pkl
│   │       ├── ensemble_model.pkl
│   │       ├── model_a_results.csv
│   │       └── model_a_nlp_metrics.csv
│   └── model_b/
│       └── traditional/
│           ├── distractor_lr_ranker.pkl
│           ├── distractor_rf_ranker.pkl
│           ├── hint_scorer.pkl
│           └── model_b_results.csv
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis (distributions, stats, etc.)
│   ├── model_training.ipynb    # Google Colab training notebook
│   └── plots/                  # Generated during training
│       ├── cm_logistic_regression.png
│       ├── cm_svm_(linearsvc).png
│       ├── cm_bernoulli_naive_bayes.png
│       ├── cm_weighted_soft-vote_ensemble.png
│       ├── cm_soft-vote_ensemble.png
│       ├── model_a_comparison.png
│       ├── model_a_nlp_metrics.png
│       ├── model_b_ranker_comparison.png
│       ├── model_b_distractor_eval.png
│       ├── model_b_hint_scorer.png
│       ├── answer_by_qtype.png
│       ├── answer_distribution.png
│       ├── article_length.png
│       ├── length_correlation.png
│       ├── option_lengths.png
│       ├── question_length.png
│       ├── question_types.png
│       └── top_words.png
│
├── src/
│   ├── preprocessing.py        # Data cleaning + OHE vectorizer training
│   ├── model_a_train.py        # Model A training (verifier + question gen)
│   ├── model_b_train.py        # Model B training (distractor + hint scorer)
│   ├── inference.py            # Unified inference API for the UI
│   └── script.py               # Quick data/model sanity check script
│
├── ui/
│   ├── app.py                  # Streamlit application (4 screens)
│   └── style.css               # Dark neon theme CSS
│
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10 or 3.11 (Python 3.14 may have compatibility issues with some packages)
- pip
- The RACE dataset CSV files (`train.csv`, `dev.csv`, `test.csv`) placed in `data/raw/`

> **Getting the RACE dataset:** Download from [https://www.cs.cmu.edu/~glai1/data/race/](https://www.cs.cmu.edu/~glai1/data/race/) or from the Hugging Face datasets hub (`datasets` library, `race` dataset, `all` config). The CSVs must have columns: `article`, `question`, `A`, `B`, `C`, `D`, `answer`.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-ML-Quiz-generation-system.git
cd AI-ML-Quiz-generation-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install streamlit==1.35.0
pip install scikit-learn==1.8.0
pip install pandas numpy scipy joblib matplotlib
pip install nltk rouge-score
```

> **Windows note:** If you see encoding errors in the terminal when running training scripts, set `PYTHONIOENCODING=utf-8` in your environment or run:
> ```bash
> set PYTHONIOENCODING=utf-8   # Windows CMD
> $env:PYTHONIOENCODING="utf-8"  # PowerShell
> ```

### 3. Verify Data

```
data/
└── raw/
    ├── train.csv   ← required
    ├── dev.csv     ← required
    └── test.csv    ← required
```

---

## 🏋️ Training Instructions

Training was done on **Google Colab (T4 GPU)**. You can run locally but it will be slower (16–120 seconds per model on the full dataset).

### Option A — Google Colab (Recommended)

1. Upload the project folder to Google Drive at:
   `My Drive/Colab Notebooks/AI PROJECT/`

2. Open `notebooks/model_training.ipynb` in Google Colab

3. Mount Drive and run all cells in order:
   - Cell 1: Mount Google Drive
   - Cell 2: Install dependencies (`scikit-learn==1.8.0`, `nltk`, `rouge-score`)
   - Cell 3: Run Model A training (`model_a_train.py`) — ~4 minutes on T4
   - Cell 4: Run Model B training (`model_b_train.py`) — ~15 minutes on T4

4. All `.pkl` model files and plot `.png` files are saved to Drive automatically.

### Option B — Local Training

**Step 1: Preprocess the data**

```bash
cd src
python preprocessing.py
```

This creates:
- `data/processed/clean_train.csv`, `clean_dev.csv`, `clean_test.csv`
- `models/onehot_encoder.pkl`
- `data/processed/train_features.npz`

**Step 2: Train Model A**

Open `src/model_a_train.py` and set at the top:
```python
is_local = True
DO_FULL_TRAIN = True   # set False for a quick test on 20k examples
```

Then run:
```bash
python model_a_train.py
```

Expected output files in `models/model_a/traditional/`:
```
lr_model.pkl, svm_model.pkl, nb_model.pkl,
kmeans_model.pkl, lp_model.pkl, ensemble_model.pkl,
model_a_results.csv, model_a_nlp_metrics.csv
```

Expected plots in `notebooks/plots/`:
```
cm_logistic_regression.png, cm_svm_(linearsvc).png,
cm_bernoulli_naive_bayes.png, cm_weighted_soft-vote_ensemble.png,
model_a_comparison.png, model_a_nlp_metrics.png
```

**Step 3: Train Model B**

Open `src/model_b_train.py` and set:
```python
is_local = True
DO_FULL_EVAL = True
```

Then run:
```bash
python model_b_train.py
```

Expected output files in `models/model_b/traditional/`:
```
distractor_lr_ranker.pkl, distractor_rf_ranker.pkl,
hint_scorer.pkl, model_b_results.csv
```

Expected plots in `notebooks/plots/`:
```
model_b_ranker_comparison.png, model_b_distractor_eval.png,
model_b_hint_scorer.png
```

### Training Time Reference (Google Colab T4)

| Step | Time |
|------|------|
| preprocessing.py | ~2 min |
| Model A — Logistic Regression | ~16 sec |
| Model A — SVM (LinearSVC + Calibration) | ~102 sec |
| Model A — Naive Bayes | ~1 sec |
| Model A — K-Means | ~30 sec |
| Model A — Label Propagation | ~7 sec |
| Model A — Ensemble | ~117 sec |
| Model B — Distractor Ranker (LR + RF) | ~23 sec |
| Model B — Hint Scorer | ~1 sec |
| Model B — Pipeline Evaluation (500 samples) | ~10 min |

---

## 🚀 Running the App

Make sure all model `.pkl` files exist in `models/` before starting the app (run training first, or copy pre-trained models).

```bash
cd ui
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

> **Important:** Run from inside the `ui/` directory so that relative paths to `src/` and `models/` resolve correctly.

If you run from the project root instead:
```bash
streamlit run ui/app.py
```

---

## 🖥 UI Screens

### Screen 1 — Article Input
- Paste any English reading passage (minimum 20 words)
- Or click one of the 4 built-in RACE sample buttons (Giant Panda, Water Cycle, Thomas Edison, Amazon Rainforest) for instant loading
- Click **Generate Quiz** to trigger both Model A (question generation) and Model B (distractor + hint generation)
- A preview of the generated question and labeled A/B/C/D options appears immediately

### Screen 2 — Take Quiz
- Displays the generated question with labeled A/B/C/D options
- User selects an answer and clicks **Check Answer**
- Model A verifies and shows confidence + latency
- Correct answers shown in green, incorrect in red
- Incorrect answers prompt navigation to the Hint Explorer

### Screen 3 — Hint Explorer
- Only active after an incorrect answer has been submitted
- Three hints are locked — unlocked one at a time:
  - **Hint 1:** General context (most vague)
  - **Hint 2:** Stronger contextual clue
  - **Hint 3:** Near-explicit passage extract (most revealing)
- **Reveal Answer** button only appears after all 3 hints are unlocked

### Screen 4 — Analytics Dashboard
- Live session KPIs: total inferences, average latency, average confidence
- Model A results table (loaded from `model_a_results.csv`)
- Model B results table + Distractor Pipeline KPIs (loaded from `model_b_results.csv`)
- All training plots rendered inline
- Full session inference log table
- **Download Session Log as CSV** button for export

---

## 📐 Evaluation Metrics

### Why two different metric sets?

The system has two distinct sub-tasks with different natures:

**Answer Verifier (classification task):** "Is this option correct or not?" is a binary classification problem, so Accuracy, F1, Precision, and Recall are the correct metrics.

**Question Generator (text generation task):** Per the instructor's note, generation tasks are evaluated with text similarity metrics rather than classification metrics:
- **BLEU-2:** Measures bigram precision between generated and reference questions. Template-based systems typically score 0.05–0.15 (neural systems reach 0.15–0.25).
- **ROUGE-L:** Measures the longest common subsequence between generated and gold questions.
- **METEOR:** Accounts for stemming and synonym matching; more lenient than BLEU.

### Why is LR accuracy only 56%?

The 75/25 class imbalance (3 wrong options per correct answer) means a model that always predicts "Wrong" gets 75% accuracy trivially — which is exactly what SVM does. LR at 56% accuracy with Macro F1 = 0.52 is actually the *best* result because it is genuinely learning both classes. The F1 on the "Correct" class (0.38) is the most meaningful metric for this task.

The theoretical ceiling for OHE bag-of-words on answer verification is limited because OHE cannot encode word order, negation, or contextual meaning — a known limitation of classical representations for NLP.

---

## ⚠️ Known Limitations

**Feature representation ceiling:** One-Hot Encoding treats "panda eats bamboo" and "bamboo eats panda" as identical feature vectors. Word order and context are lost, which limits verifier accuracy regardless of classifier choice.

**Class imbalance:** 4-choice MCQ datasets are inherently 75% "wrong" by construction. This is a structural property of the RACE dataset, not a bug.

**Distractor quality:** Template-generated distractors are phrase-level chunks from the passage rather than semantically crafted alternatives. They achieve high diversity (0.97) but low overlap with gold RACE distractors (precision ~0.05). This is expected and honest for a classical ML system.

**Question generation:** Template-based generation with Wh-word substitution produces grammatically simple questions. BLEU-2 of 0.05 reflects that generated questions differ in phrasing from gold RACE questions, though they are semantically valid comprehension questions about the passage.

---

## 📄 License

This project was developed for academic purposes as part of the AL2002 Artificial Intelligence Lab course at FAST NUCES Islamabad. The RACE dataset is subject to its own license and terms of use.
