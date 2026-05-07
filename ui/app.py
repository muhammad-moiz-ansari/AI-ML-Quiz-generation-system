import streamlit as st
import os
import sys
import pandas as pd
import time

# ── 1. Setup & Imports ───────────────────────────────────────────────────────
# We must add the root directory to the system path so we can import 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from src.inference import verify_answer, generate_question, get_session_metrics

# Page config MUST be the very first Streamlit command
st.set_page_config(page_title="Neon AI | Reading Comprehension", page_icon="⚡", layout="wide")

# Load Custom Neon CSS
def load_css():
    css_path = os.path.join(current_dir, 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ `style.css` not found in the `ui/` folder. Neon theme might not apply fully.")

load_css()

# ── 2. Session State Initialization ──────────────────────────────────────────
if 'article' not in st.session_state:
    st.session_state.article = ""
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# ── 3. Sidebar Navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: var(--neon-blue);'>⚡ NEON AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### Navigation")
    page = st.radio("Go to:", [
        "1. 📖 Article Input", 
        "2. 🎯 Take Quiz", 
        "3. 💡 Hint Explorer", 
        "4. 📊 Analytics Dashboard"
    ])
    
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Model A: Loaded ✅<br>Model B: Loaded ✅</div>", unsafe_allow_html=True)

# ── 4. Main Application Screens ──────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
# SCREEN 1: ARTICLE INPUT
# ════════════════════════════════════════════════════════════
if page == "1. 📖 Article Input":
    st.title("Step 1: Provide Knowledge Base")
    st.markdown("Paste an article or reading passage below. The AI will generate a comprehension quiz based on this text.")
    
    article_text = st.text_area("Passage Text", height=250, value=st.session_state.article,
                                placeholder="Once upon a time in the world of machine learning...")
    
    if st.button("🚀 Generate AI Quiz", type="primary"):
        if len(article_text.split()) < 20:
            st.error("❌ Passage is too short! Please provide at least 20 words.")
        else:
            with st.spinner("⚡ AI is reading the passage and generating questions..."):
                # Save to state
                st.session_state.article = article_text
                # Call inference.py Model A generator
                quiz = generate_question(article_text)
                st.session_state.quiz_data = quiz
                st.session_state.last_result = None # Reset previous answers
                
                time.sleep(1) # Visual effect
                st.success("✅ Quiz generated successfully! Go to the '🎯 Take Quiz' tab from the sidebar.")

# ════════════════════════════════════════════════════════════
# SCREEN 2: TAKE QUIZ
# ════════════════════════════════════════════════════════════
elif page == "2. 🎯 Take Quiz":
    st.title("Step 2: Test Your Comprehension")
    
    if not st.session_state.quiz_data:
        st.warning("⚠️ No quiz found! Please go to 'Article Input' and generate a quiz first.")
    else:
        quiz = st.session_state.quiz_data
        
        # Display the Article in a collapsible box so they can refer to it
        with st.expander("📄 View Reference Passage"):
            st.write(st.session_state.article)
            
        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        
        # Display Question
        st.markdown(f"### ❓ {quiz['question']}")
        
        # Format options for the radio button
        options_list = list(quiz['options'].values())
        
        selected_option = st.radio("Choose your answer:", options_list, index=None)
        
        if st.button("Check Answer ⚡", type="primary"):
            if selected_option is None:
                st.error("Please select an answer first!")
            else:
                with st.spinner("Verifying with Model A Ensemble..."):
                    # Call inference.py verify_answer
                    result = verify_answer(
                        article=st.session_state.article,
                        question=quiz['question'],
                        option=selected_option
                    )
                    st.session_state.last_result = result
                    st.session_state.last_selected = selected_option
        
        # Display Results if answered
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
            
            if res['correct']:
                st.balloons()
                st.success(f"🎉 CORRECT! Model Confidence: {res['confidence']*100:.1f}%")
            else:
                st.error(f"❌ INCORRECT! Model Confidence that this is correct: {res['confidence']*100:.1f}%")
                st.info("💡 Tip: Go to the 'Hint Explorer' tab to see why this distractor tricked you!")
            
            st.caption(f"Inference Latency: {res['latency_ms']}ms | AI Verifier: {res['model_used']}")

# ════════════════════════════════════════════════════════════
# SCREEN 3: HINT EXPLORER (Model B Integration)
# ════════════════════════════════════════════════════════════
elif page == "3. 💡 Hint Explorer":
    st.title("Step 3: AI Learning Hints")
    
    if not st.session_state.last_result:
        st.warning("⚠️ Take the quiz first to generate hints based on your answer.")
    elif st.session_state.last_result['correct']:
        st.success("✅ You got the answer right! No hints needed. Great job!")
    else:
        st.markdown("### Why was your answer wrong?")
        st.write(f"**Your Choice:** {st.session_state.last_selected}")
        
        # Simulated Model B Output (Since Model B isn't fully in inference.py yet)
        # You will connect your partner's actual Model B Hint function here later!
        st.markdown(
            """
            <div class='info-box' style='border-color: var(--neon-pink); color: #E2E8F0;'>
                <strong style='color: var(--neon-pink);'>Model B Analysis:</strong><br><br>
                This is a <em>Plausible Distractor</em>. It uses words found in the text to trick you, 
                but it alters the core meaning. Look closely at the context of the sentence in the passage again.
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SCREEN 4: ANALYTICS DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "4. 📊 Analytics Dashboard":
    st.title("System Analytics & Telemetry")
    
    # Get live stats from inference.py
    metrics = get_session_metrics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Inferences", value=metrics['total_inferences'])
    with col2:
        st.metric(label="Avg Latency (ms)", value=f"{metrics['avg_latency_ms']} ms")
    with col3:
        avg_conf = metrics['model_a_stats'].get('avg_confidence', 0)
        st.metric(label="Avg AI Confidence", value=f"{avg_conf*100:.1f}%")
        
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    
    # Live Session Log Table
    st.subheader("Live Inference Log")
    if metrics['session_log']:
        df_log = pd.DataFrame(metrics['session_log'])
        # Reorder columns to look nice
        cols = ['timestamp', 'task', 'model_used', 'latency_ms', 'predicted', 'confidence']
        existing_cols = [c for c in cols if c in df_log.columns]
        st.dataframe(df_log[existing_cols], use_container_width=True)
    else:
        st.write("No inferences made in this session yet.")
        
    st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
    
    # Model Training Plots (From your EDA/Training)
    st.subheader("Model Training Insights")
    st.markdown("These plots were generated during the AI training phase on the FAST-NUCES server.")
    
    plot_col1, plot_col2 = st.columns(2)
    
    plots_dir = os.path.join(root_dir, 'notebooks', 'plots')
    
    with plot_col1:
        cm_path = os.path.join(plots_dir, 'cm_soft-vote_ensemble.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Ensemble Confusion Matrix")
        else:
            st.info("Confusion matrix plot not found. Run model_a_train.py!")
            
    with plot_col2:
        comp_path = os.path.join(plots_dir, 'model_a_comparison.png')
        if os.path.exists(comp_path):
            st.image(comp_path, caption="Model Performance Comparison")
        else:
            st.info("Comparison chart not found. Run model_a_train.py!")