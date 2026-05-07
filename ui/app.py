import streamlit as st
import os
import sys
import pandas as pd
import io
import time
import random

# ── 1. Setup & Imports ───────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from src.inference import (
    verify_answer, generate_question,
    get_session_metrics, get_hints, get_model_file_metrics
)

# Page config MUST be the very first Streamlit command
st.set_page_config(
    page_title="AI Reading Comprehension System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Custom Neon CSS ──────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(current_dir, 'style.css')
    if os.path.exists(css_path):
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found in the ui/ folder. Neon theme might not apply fully.")

load_css()

# ── 2. Session State Initialization ──────────────────────────────────────────
defaults = {
    'article'          : "",
    'quiz_data'        : None,
    'last_result'      : None,
    'last_selected'    : None,
    'is_correct'       : None,   # ground-truth correctness (not AI guess)
    'hints_unlocked'   : 0,      # how many hints the user has unlocked (0-3)
    'answer_revealed'  : False,  # True once user clicks "Reveal Answer"
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── RACE sample passages for quick testing ───────────────────────────────────
RACE_SAMPLES = [
    {
        "title": "The Giant Panda",
        "text": (
            "The giant panda is a bear native to South Central China. "
            "It is characterised by its bold black-and-white coat and large body. "
            "Though it belongs to the order Carnivora, the giant panda's diet is over "
            "99 percent bamboo. Pandas spend 10 to 16 hours each day eating bamboo "
            "shoots, leaves, and stems. The giant panda lives in a few mountain ranges "
            "in central China, mainly in Sichuan province. It is a national symbol of "
            "China and is used in the logo of the World Wildlife Fund. "
            "As of 2016, about 1,864 giant pandas live in the wild."
        )
    },
    {
        "title": "The Water Cycle",
        "text": (
            "The water cycle, also known as the hydrological cycle, describes the "
            "continuous movement of water on, above, and below the surface of the Earth. "
            "The sun drives the water cycle by heating water in rivers, lakes, and oceans, "
            "causing it to evaporate into water vapour. This vapour rises into the atmosphere "
            "where it cools and condenses into clouds. When clouds become heavy with water "
            "droplets, precipitation occurs in the form of rain or snow. "
            "Water then flows across the land as surface runoff, collecting in streams and "
            "rivers that eventually return to the ocean, completing the cycle. "
            "Groundwater also plays an important role, slowly filtering through soil and rock."
        )
    },
    {
        "title": "Thomas Edison",
        "text": (
            "Thomas Alva Edison was an American inventor and businessman who developed "
            "many devices that greatly influenced life around the world. He is often "
            "described as America's greatest inventor. He developed a system of electrical "
            "power generation and distribution to homes, businesses, and factories. "
            "Edison's most famous invention is the long-lasting, practical electric light bulb, "
            "which he demonstrated in 1879. He also invented the phonograph in 1877, "
            "which was the first device to record and play back sound. "
            "Edison's laboratory in Menlo Park, New Jersey became the world's first "
            "industrial research laboratory. He held over 1,000 patents for his inventions."
        )
    },
    {
        "title": "The Amazon Rainforest",
        "text": (
            "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical "
            "rainforest in the Amazon biome that covers most of the Amazon basin of South America. "
            "This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the "
            "rainforest. The Amazon represents over half of the planet's remaining rainforests, "
            "and comprises the largest and most biodiverse tract of tropical rainforest in the world. "
            "The forest contains an estimated 390 billion individual trees divided into "
            "16,000 species. It is home to about 10 percent of all species on Earth. "
            "Scientists estimate that a single hectare of Amazon rainforest can contain "
            "more species of insects than the entire British Isles."
        )
    },
]

# ── 3. Sidebar Navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h1 style='text-align:center;color:var(--neon-blue);'>⚡ NEON AI</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio("Go to:", [
        "1. 📖 Article Input",
        "2. 🎯 Take Quiz",
        "3. 💡 Hint Explorer",
        "4. 📊 Analytics Dashboard",
    ])

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Model A: Loaded ✅<br>Model B: Loaded ✅</div>",
        unsafe_allow_html=True
    )

    # Quick status summary
    if st.session_state.quiz_data:
        q_preview = st.session_state.quiz_data['question'][:50] + "..."
        st.markdown(
            f"<div class='info-box' style='margin-top:0.6rem;'>"
            f"Active quiz:<br><em>{q_preview}</em></div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — ARTICLE INPUT
# ════════════════════════════════════════════════════════════════════════════
if page == "1. 📖 Article Input":
    st.title("Step 1: Provide a Reading Passage")
    st.markdown(
        "Paste any article below, or load a quick sample from our built-in "
        "RACE-style passages. Then click **Generate Quiz** to run both Model A "
        "(question generator) and Model B (distractor + hint generator)."
    )

    # ── Load Random RACE Sample ───────────────────────────────────────────
    st.markdown("#### Quick Load — RACE Sample Passage")
    sample_cols = st.columns(len(RACE_SAMPLES))
    for i, sample in enumerate(RACE_SAMPLES):
        with sample_cols[i]:
            if st.button(f"📄 {sample['title']}", use_container_width=True):
                st.session_state.article    = sample['text']
                st.session_state.quiz_data  = None
                st.session_state.last_result = None
                st.session_state.is_correct  = None
                st.session_state.hints_unlocked  = 0
                st.session_state.answer_revealed = False
                st.rerun()

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Text Area ─────────────────────────────────────────────────────────
    article_text = st.text_area(
        "Passage Text",
        height=280,
        value=st.session_state.article,
        placeholder="Paste your reading passage here (minimum 20 words)..."
    )

    word_count = len(article_text.split()) if article_text.strip() else 0
    st.caption(f"Word count: {word_count}")

    # ── Submit Button — triggers BOTH Model A and Model B ─────────────────
    if st.button("🚀 Generate Quiz (Model A + Model B)", type="primary", use_container_width=True):
        if word_count < 20:
            st.error("Passage is too short! Please provide at least 20 words.")
        else:
            with st.spinner("Model A is generating a question... Model B is building distractors and hints..."):
                st.session_state.article         = article_text
                quiz = generate_question(article_text)
                st.session_state.quiz_data       = quiz
                st.session_state.last_result     = None
                st.session_state.last_selected   = None
                st.session_state.is_correct      = None
                st.session_state.hints_unlocked  = 0
                st.session_state.answer_revealed = False
                time.sleep(0.8)

            st.success(
                "Quiz generated! "
                f"Question: *{quiz['question'][:80]}...* — "
                "Go to **Take Quiz** to answer."
            )

            # Preview the generated question + options on Screen 1
            st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
            st.markdown("#### Preview — Generated Question")
            st.markdown(f"**{quiz['question']}**")
            for letter, text in quiz['options'].items():
                marker = " ✅" if letter == quiz['correct_label'] else ""
                #st.markdown(f"- **{letter}.** {text}{marker}")
                st.markdown(f"- **{letter}.** {text}")
            st.caption(f"Source: {quiz.get('source', 'Template generated')}")


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — QUESTION & ANSWER QUIZ VIEW
# ════════════════════════════════════════════════════════════════════════════
elif page == "2. 🎯 Take Quiz":
    st.title("Step 2: Test Your Comprehension")

    if not st.session_state.quiz_data:
        st.warning("No quiz found! Go to Article Input and generate a quiz first.")
    else:
        quiz = st.session_state.quiz_data

        with st.expander("📄 View Reference Passage"):
            st.write(st.session_state.article)

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

        # ── Question ──────────────────────────────────────────────────────
        st.markdown(
            f"<div class='question-text'>❓ {quiz['question']}</div>",
            unsafe_allow_html=True
        )

        # ── Options — labeled A / B / C / D ──────────────────────────────
        st.markdown("**Choose your answer:**")
        options_display = [
            f"{letter}. {text}"
            for letter, text in quiz['options'].items()
        ]
        # Map display string back to raw option text for grading
        display_to_text = {
            f"{letter}. {text}": text
            for letter, text in quiz['options'].items()
        }

        selected_display = st.radio(
            "Options",
            options=options_display,
            index=None,
            label_visibility="collapsed"
        )

        if st.button("Check Answer ⚡", type="primary"):
            if selected_display is None:
                st.error("Please select an answer first!")
            else:
                selected_text = display_to_text[selected_display]

                with st.spinner("Model A is verifying your answer..."):
                    res = verify_answer(
                        article=st.session_state.article,
                        question=quiz['question'],
                        option=selected_text
                    )

                st.session_state.last_result   = res
                st.session_state.last_selected = selected_text

                # Ground-truth correctness: compare against the known correct answer
                is_correct = (selected_text == quiz['correct_text'])
                st.session_state.is_correct = is_correct

                # Reset hints when a new answer is submitted
                st.session_state.hints_unlocked  = 0
                st.session_state.answer_revealed = False

                st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

                if is_correct:
                    st.balloons()
                    st.markdown(
                        "<div class='result-correct'>"
                        "🎉 CORRECT! Well done — your answer matches the passage."
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='result-wrong'>"
                        f"❌ INCORRECT. The correct answer was: "
                        f"<strong>{quiz['correct_text']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    st.info("Go to the **Hint Explorer** tab to get graduated clues and try again.")

                # Model A's probabilistic guess (informational)
                ai_label = "Correct" if res['correct'] else "Wrong"
                st.caption(
                    f"Model A Guess: **{ai_label}** | "
                    f"Confidence: {res['confidence']*100:.1f}% | "
                    f"Model: {res['model_used']} | "
                    f"Latency: {res['latency_ms']} ms"
                )

        # ── Show previous result if already answered ──────────────────────
        elif st.session_state.last_result is not None:
            st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
            if st.session_state.is_correct:
                st.markdown(
                    "<div class='result-correct'>Last answer: CORRECT</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-wrong'>"
                    f"Last answer: INCORRECT — correct was "
                    f"<strong>{quiz['correct_text']}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — HINT PANEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "3. 💡 Hint Explorer":
    st.title("Step 3: Graduated Hints")

    if not st.session_state.quiz_data:
        st.warning("No quiz active. Go to Article Input and generate a quiz first.")

    elif st.session_state.is_correct is None:
        # User has not attempted the quiz yet
        st.info("Answer the quiz on the **Take Quiz** tab first, then come back for hints.")

    elif st.session_state.is_correct:
        # User was correct — no hints needed
        st.markdown(
            "<div class='result-correct'>"
            "You answered correctly! No hints needed. Great work."
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Correct answer:** {st.session_state.quiz_data['correct_text']}")

    else:
        # User was wrong — show graduated hints
        st.markdown(f"**Your answer:** {st.session_state.last_selected}")
        st.markdown(
            f"**Correct answer:** *(hidden — use hints to find it)*"
        )

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Model B — Graduated Hints")
        st.caption(
            "Model B's Hint Scorer has ranked passage sentences by relevance to "
            "the question. Unlock hints one at a time — Hint 1 is vague, "
            "Hint 3 is near-explicit."
        )

        # Generate hints once (cache in session)
        if 'cached_hints' not in st.session_state or st.session_state.get('hints_question') != st.session_state.quiz_data['question']:
            with st.spinner("Model B is scoring passage sentences..."):
                st.session_state.cached_hints   = get_hints(
                    st.session_state.article,
                    st.session_state.quiz_data['question']
                )
                st.session_state.hints_question = st.session_state.quiz_data['question']

        hints = st.session_state.cached_hints

        # ── Hint 1 ────────────────────────────────────────────────────────
        with st.expander("💡 Hint 1 — General Context", expanded=(st.session_state.hints_unlocked >= 1)):
            if st.session_state.hints_unlocked >= 1:
                st.info(hints[0])
            else:
                st.caption("🔒 Click the button below to unlock this hint.")

        # ── Hint 2 ────────────────────────────────────────────────────────
        with st.expander("🔍 Hint 2 — Stronger Clue", expanded=(st.session_state.hints_unlocked >= 2)):
            if st.session_state.hints_unlocked >= 2:
                st.warning(hints[1])
            else:
                st.caption("🔒 Unlock Hint 1 first.")

        # ── Hint 3 ────────────────────────────────────────────────────────
        with st.expander("🎯 Hint 3 — Near-Explicit Extract", expanded=(st.session_state.hints_unlocked >= 3)):
            if st.session_state.hints_unlocked >= 3:
                st.error(hints[2])
            else:
                st.caption("🔒 Unlock Hint 2 first.")

        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

        # ── Unlock buttons ────────────────────────────────────────────────
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.session_state.hints_unlocked < 1:
                if st.button("Unlock Hint 1", use_container_width=True):
                    st.session_state.hints_unlocked = 1
                    st.rerun()
            else:
                st.success("Hint 1 unlocked")

        with btn_col2:
            if st.session_state.hints_unlocked == 1:
                if st.button("Unlock Hint 2", use_container_width=True):
                    st.session_state.hints_unlocked = 2
                    st.rerun()
            elif st.session_state.hints_unlocked >= 2:
                st.success("Hint 2 unlocked")
            else:
                st.button("Unlock Hint 2", disabled=True, use_container_width=True)

        with btn_col3:
            if st.session_state.hints_unlocked == 2:
                if st.button("Unlock Hint 3", use_container_width=True):
                    st.session_state.hints_unlocked = 3
                    st.rerun()
            elif st.session_state.hints_unlocked >= 3:
                st.success("Hint 3 unlocked")
            else:
                st.button("Unlock Hint 3", disabled=True, use_container_width=True)

        # ── Reveal Answer — only after all 3 hints are unlocked ──────────
        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

        if st.session_state.hints_unlocked < 3:
            st.caption("Unlock all 3 hints to reveal the answer.")
            st.button("Reveal Answer", disabled=True, use_container_width=True)
        else:
            if not st.session_state.answer_revealed:
                if st.button("🎯 Reveal Answer", type="primary", use_container_width=True):
                    st.session_state.answer_revealed = True
                    st.rerun()
            else:
                st.markdown(
                    f"<div class='result-correct'>"
                    f"The correct answer is: <strong>{st.session_state.quiz_data['correct_text']}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — DEVELOPER / ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "4. 📊 Analytics Dashboard":
    st.title("Developer Analytics Dashboard")
    st.caption("Live session telemetry + saved training metrics from Model A and Model B.")

    metrics     = get_session_metrics()
    file_metrics = get_model_file_metrics()

    # ── Row 1: Live session KPIs ──────────────────────────────────────────
    st.markdown("### Live Session Metrics")
    k1, k2, k3, k4 = st.columns(4)

    total_v = len([r for r in metrics['session_log'] if r.get('task') == 'verify_answer'])
    correct_v = len([
        r for r in metrics['session_log']
        if r.get('task') == 'verify_answer' and r.get('predicted') == 'Correct'
    ])
    session_acc = (correct_v / total_v * 100) if total_v > 0 else 0.0
    avg_conf    = metrics['model_a_stats'].get('avg_confidence', 0)

    with k1:
        st.metric("Total Inferences", metrics['total_inferences'])
    with k2:
        st.metric("Avg Latency (ms)", f"{metrics['avg_latency_ms']:.1f}")
    with k3:
        st.metric("Avg AI Confidence", f"{avg_conf*100:.1f}%")
    with k4:
        st.metric("Session Predictions", f"{total_v} verifications")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Row 2: Model A Training Results (from CSV) ────────────────────────
    st.markdown("### Model A — Training Results (Dev Set)")

    if "error" not in file_metrics and "model_a_results" in file_metrics:
        df_a = pd.DataFrame(file_metrics["model_a_results"])
        df_a.columns = [c.replace('_', ' ').title() for c in df_a.columns]

        # Colour the best row
        st.dataframe(df_a, use_container_width=True)

        # Highlight the best model
        if 'Macro F1' in df_a.columns:
            best_row = df_a.loc[df_a['Macro F1'].astype(float).idxmax()]
            st.caption(
                f"Best Model A: **{best_row['Model']}** "
                f"(Accuracy: {float(best_row['Accuracy']):.4f}, "
                f"Macro F1: {float(best_row['Macro F1']):.4f})"
            )
    else:
        st.info("model_a_results.csv not found. Run model_a_train.py to generate it.")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Row 3: Model B Training Results (from CSV) ────────────────────────
    st.markdown("### Model B — Training Results (Distractor & Hint Scorer)")

    model_b_results_path = os.path.join(root_dir, "models", "model_b", "traditional", "model_b_results.csv")
    if os.path.exists(model_b_results_path):
        df_b = pd.read_csv(model_b_results_path)
        st.dataframe(df_b, use_container_width=True)

        # Show key distractor pipeline metrics as KPIs
        pipeline_row = df_b[df_b['component'].str.contains('Pipeline', na=False)]
        if not pipeline_row.empty:
            r = pipeline_row.iloc[0]
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Distractor Precision", f"{float(r.get('Precision', 0)):.4f}")
            with b2:
                st.metric("Distractor Recall", f"{float(r.get('Recall', 0)):.4f}")
            with b3:
                st.metric("Distractor F1", f"{float(r.get('F1', 0)):.4f}")
            with b4:
                st.metric("Distractor Diversity", f"{float(r.get('Diversity', 0)):.4f}")
    else:
        st.info("model_b_results.csv not found. Run model_b_train.py to generate it.")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Row 4: Training Plots ─────────────────────────────────────────────
    st.markdown("### Model Training Plots")
    plots_dir = os.path.join(root_dir, 'notebooks', 'plots')

    plot_files = {
        'cm_soft-vote_ensemble.png'    : "Model A — Ensemble Confusion Matrix",
        'model_a_comparison.png'       : "Model A — Supervised Comparison",
        'model_b_ranker_comparison.png': "Model B — Ranker Comparison",
        'model_b_distractor_eval.png'  : "Model B — Distractor Pipeline Eval",
    }

    plot_cols = st.columns(2)
    col_idx   = 0
    found_any = False
    for filename, caption in plot_files.items():
        path = os.path.join(plots_dir, filename)
        if os.path.exists(path):
            with plot_cols[col_idx % 2]:
                st.image(path, caption=caption, use_container_width=True)
            col_idx  += 1
            found_any = True

    if not found_any:
        st.info("No training plots found. Run model_a_train.py and model_b_train.py to generate them.")

    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

    # ── Row 5: Live Inference Log + CSV Export ────────────────────────────
    st.markdown("### Live Inference Log")

    if metrics['session_log']:
        df_log = pd.DataFrame(metrics['session_log'])
        cols_order = ['timestamp', 'task', 'model_used', 'latency_ms', 'predicted', 'confidence']
        existing   = [c for c in cols_order if c in df_log.columns]
        st.dataframe(df_log[existing], use_container_width=True)

        # ── CSV Export button ─────────────────────────────────────────────
        csv_buffer = io.StringIO()
        df_log[existing].to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')

        st.download_button(
            label="Download Session Log as CSV",
            data=csv_bytes,
            file_name=f"session_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.write("No inferences made in this session yet. Generate a quiz and answer it first.")