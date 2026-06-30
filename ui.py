import streamlit as st
import os
import time
from backend.transcribe import transcribe_audio
from backend.chunking import create_chunks
from backend.embedding import generate_embeddings, save_embeddings_and_metadata
from backend.faiss_index import build_faiss_index, save_faiss_index
from backend.retrieval import search_query
from backend.summarizer import summarize_transcript
from backend.quiz import generate_quiz_data
import base64

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="InSight Audio | AI Podcast Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# 🎨 PREMIERE DESIGN SYSTEM (CSS)
# -----------------------------------------------------
st.markdown("""
<style>
    /* ----- FONTS ----- */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --background: #0f172a;
        --surface: #1e293b;
        --text: #f8fafc;
        --text-muted: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ----- BACKGROUND & MAIN LAYOUT ----- */
    .stApp {
        background-color: var(--background);
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* ----- SIDEBAR ----- */
    [data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        margin-bottom: 2rem;
    }
    .logo-insight { color: #1e3a8a; font-weight: 800; }
    .logo-audio { color: #64748b; font-weight: 300; }

    /* Navigation Radio Buttons - Styled as Pills */
    .stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    
    .stRadio label {
        background-color: transparent;
        padding: 10px 15px;
        border-radius: 8px;
        transition: all 0.2s ease;
        color: var(--text-muted);
        font-weight: 500;
        border: 1px solid transparent;
        cursor: pointer;
    }
    
    .stRadio label:hover {
        background-color: rgba(255,255,255,0.03);
        color: white;
    }

    div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    /* ----- TITLES & HEADERS ----- */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-muted);
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* ----- CARDS & CONTAINERS ----- */
    .feature-card {
        background-color: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    .result-box {
        background: linear-gradient(145deg, rgba(30,41,59,0.8), rgba(15,23,42,0.8));
        border-left: 4px solid var(--primary);
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }

    /* ----- BUTTONS ----- */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
    }

    /* ----- INPUT FIELDS ----- */
    .stTextInput > div > div > input {
        background-color: rgba(2, 6, 23, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px;
        padding: 12px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* ----- METRICS ----- */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700;
        background: -webkit-linear-gradient(#eee, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Sidebar Navigation with State Management
# -----------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "active_file" not in st.session_state:
    st.session_state.active_file = "None"
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False

with st.sidebar:
    st.markdown("""
<div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 2rem;">
    <svg width="100%" height="auto" viewBox="0 0 280 60" style="max-width: 280px;" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="playGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#4e7fae" />
                <stop offset="100%" stop-color="#10355e" />
            </linearGradient>
        </defs>
        <path d="M 5 12 Q 5 7 11 10 L 45 27 Q 50 30 45 33 L 11 50 Q 5 53 5 48 Z" fill="url(#playGrad)" />
        <circle cx="34" cy="30" r="18" fill="white" stroke="#10355e" stroke-width="3" />
        <line x1="47" y1="43" x2="56" y2="55" stroke="#10355e" stroke-width="5" stroke-linecap="round" />
        <line x1="22" y1="30" x2="22" y2="30" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <line x1="25" y1="26" x2="25" y2="34" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <line x1="28" y1="22" x2="28" y2="38" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <line x1="40" y1="22" x2="40" y2="38" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <line x1="43" y1="26" x2="43" y2="34" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <line x1="46" y1="30" x2="46" y2="30" stroke="#10355e" stroke-width="2" stroke-linecap="round" />
        <text x="34" y="36.5" font-family="'Plus Jakarta Sans', Arial, sans-serif" font-weight="800" font-size="20" fill="#10355e" text-anchor="middle">?</text>
        <text x="68" y="38" font-family="'Plus Jakarta Sans', sans-serif" font-size="28">
            <tspan font-weight="800" fill="#10355e">InSight</tspan>
            <tspan font-weight="400" fill="#647e96" dx="6">Audio</tspan>
        </text>
    </svg>
</div>
    """, unsafe_allow_html=True)
    
    # Navigation Radio Button tied to session state
    selected_page = st.radio(
        "Navigate",
        ["Dashboard", "Upload", "Search", "Summary", "Quiz"],
        index=["Dashboard", "Upload", "Search", "Summary", "Quiz"].index(st.session_state.page),
        # key="nav_radio",  <-- Removed to prevent key modification error
        label_visibility="collapsed"
    )
    
    # Update session state when radio button changes
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()
    
    st.markdown("---")
    
    # Status Indicator
    file_name = st.session_state.active_file
    if st.session_state.is_processed:
        status_text = "● Context Loaded"
        status_color = "#4ade80" # Green
    elif file_name != "None":
        status_text = "● Pending Processing"
        status_color = "#fbbf24" # Yellow
    else:
        status_text = "● Waiting for Upload"
        status_color = "#94a3b8" # Gray

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px;">
        <p style="margin:0; font-size: 0.8rem; color: #94a3b8; font-weight: 600;">ACTIVE PODCAST</p>
        <p style="margin:5px 0 0 0; font-size: 0.9rem; color: white; word-wrap: break-word; overflow-wrap: break-word;">{file_name}</p>
        <p style="margin:2px 0 0 0; font-size: 0.7rem; color: {status_color};">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------
# MAIN: DASHBOARD (Home)
# -----------------------------------------------------
if st.session_state.page == "Dashboard":
    st.markdown('<h1 class="main-header">Your Audio, Intelligent.</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Unlock insights from your podcasts with AI-powered search, summarization, and active learning tools.</p>', unsafe_allow_html=True)
    
    # Feature Grid
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin-top:0">🚀 Upload</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Ingest audio files (MP3, WAV) specifically optimized for speech recognition.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin-top:0">🧠 Search</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Semantic vector search allows you to find concepts, not just keywords.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3 style="margin-top:0">🎓 Quiz</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">Generate dynamic MCQs to test your retention of the podcast content.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    # Working Buttons
    with col1:
        if st.button("📂 Upload New"):
            st.session_state.page = "Upload"
            st.rerun()
    with col2:
        if st.button("❓ Start Quiz"):
            st.session_state.page = "Quiz"
            st.rerun()

# -----------------------------------------------------
# PAGE: UPLOAD
# -----------------------------------------------------
elif st.session_state.page == "Upload":
    st.markdown('<h1 class="main-header">Ingest Media</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your podcast episode to the processing pipeline.</p>', unsafe_allow_html=True)

    # Removed the broken blank rectangle div
    # Replaced with a useful info box
    st.info("💡 **Tip:** For best results, ensure your audio is clear and under 200MB. Supported formats: .mp3, .wav, .m4a.")

    uploaded_file = st.file_uploader("Select Audio File", type=["wav", "mp3", "m4a"], key="main_upload")

    # Handle New Uploads and save them
    if uploaded_file and st.session_state.active_file != uploaded_file.name:
        st.session_state.active_file = uploaded_file.name
        st.session_state.is_processed = False
        
        os.makedirs("data/raw_audio", exist_ok=True)
        save_path = os.path.join("data/raw_audio", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.rerun()

    # Always show controls if a file is active
    if st.session_state.active_file != "None":
        if st.session_state.is_processed:
            st.success(f"File `{st.session_state.active_file}` has been processed and is ready!")
        else:
            st.success(f"File `{st.session_state.active_file}` staged for processing.")
        
        st.markdown("### ⚙️ Pipeline Control")
        
        if st.button("Run Indexing Sequence", disabled=st.session_state.is_processed):
            save_path = os.path.join("data/raw_audio", st.session_state.active_file)
            with st.status("Processing Audio Pipeline...", expanded=True) as status:
                st.write("🎙️ Extracting Audio & Transcribing...")
                transcript = transcribe_audio(save_path, "data/transcripts/test_transcript.json")
                
                st.write("✂️ Segmenting text into semantic chunks...")
                chunks = create_chunks(transcript, max_words=200)
                
                st.write("🧠 Generating Vector Embeddings...")
                embeddings = generate_embeddings(chunks)
                save_embeddings_and_metadata(embeddings, chunks)
                
                st.write("🗂️ Building FAISS Index...")
                index = build_faiss_index(embeddings)
                save_faiss_index(index)
                
                st.session_state.quiz_data = [] # Reset quiz
                st.session_state.is_processed = True
                status.update(label="Indexing Complete!", state="complete", expanded=False)
            
            st.balloons()
            st.success("Podcast is now live and searchable!")
            st.rerun()

# -----------------------------------------------------
# PAGE: SEARCH
# -----------------------------------------------------
elif st.session_state.page == "Search":
    st.markdown('<h1 class="main-header">Deep Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask natural language questions to retrieve precise audio segments.</p>', unsafe_allow_html=True)

    query = st.text_input("Query", placeholder="e.g. What is the impact of AI on creative jobs?", label_visibility="collapsed")
    
    if query:
        st.write("")
        with st.spinner("Scanning vector database..."):
            result = search_query(query, top_k=3)
            
            if result:
                 # Custom Result Layout
                st.markdown(f"""
                <div class="result-box">
                    <h3 style="margin-top:0; color:#fff;">💡 Answer</h3>
                    <p style="font-size: 1.1rem; line-height: 1.6; color: #cbd5e1;">{result['answer']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics Row
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Confidence", f"{result['similarity_score']:.2f}")
                with m2:
                    st.metric("Timestamp", f"{int(result['start'])}s")
                with m3:
                    st.metric("Duration", f"{int(result['end'] - result['start'])}s")
                
                with st.expander("Show Evidence (Raw Transcript)"):
                    st.info(f"\"{result['source_text']}\"")
            else:
                 st.warning("No relevant answer found in the transcript.")

# -----------------------------------------------------
# PAGE: SUMMARY
# -----------------------------------------------------
elif st.session_state.page == "Summary":
    st.markdown('<h1 class="main-header">Executive Brief</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">High-level overview of the entire episode.</p>', unsafe_allow_html=True)

    if st.button("Generate Summary"):
        with st.spinner("Synthesizing transcript..."):
            summary = summarize_transcript()
            st.markdown(f"""
            <div class="feature-card">
                <p style="font-size: 1.05rem; line-height: 1.8; color: #e2e8f0;">{summary}</p>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------
# PAGE: QUIZ
# -----------------------------------------------------
elif st.session_state.page == "Quiz":
    st.markdown('<h1 class="main-header">Active Recall</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test your understanding with an AI-generated assessment.</p>', unsafe_allow_html=True)

    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = []
    
    # Session state to track current question index
    if "quiz_idx" not in st.session_state:
        st.session_state.quiz_idx = 0
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_complete" not in st.session_state:
        st.session_state.quiz_complete = False

    if not st.session_state.quiz_data:
        col1, col2 = st.columns([1,2])
        with col1:
             if st.button("Create New Quiz"):
                with st.spinner("Generating questions..."):
                    transcript_path = "data/transcripts/test_transcript.json"
                    if os.path.exists(transcript_path):
                        data = generate_quiz_data(transcript_path)
                        st.session_state.quiz_data = data
                        st.session_state.quiz_idx = 0
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_complete = False
                        st.rerun()
                    else:
                        st.error("No transcript found. Please upload first.")
    
    else:
        # Display Quiz One-by-One
        q_data = st.session_state.quiz_data
        idx = st.session_state.quiz_idx
        
        if not st.session_state.quiz_complete:
            # Progress bar
            progress = (idx / len(q_data))
            st.progress(progress, text=f"Question {idx+1} of {len(q_data)}")
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            
            current_q = q_data[idx]
            st.markdown(f"### {idx+1}. {current_q['question']}")
            
            # Using a unique key per question so selection resets!
            answer = st.radio(
                "Select an Option:",
                ["A", "B", "C", "D"],
                format_func=lambda x: f"{x}) {current_q['options'].get(x, '')}",
                key=f"q_{idx}",
                index=None # No default selection
            )
            
            st.write("")
            
            if st.button("Submit Answer"):
                if answer:
                    if answer == current_q['correct_answer']:
                        st.success("Correct! ✅")
                        st.session_state.quiz_score += 1
                    else:
                        st.error(f"Incorrect ❌. Correct answer: {current_q['correct_answer']}")
                    
                    time.sleep(1) # Pause to let user see result
                    
                    if idx + 1 < len(q_data):
                         st.session_state.quiz_idx += 1
                         st.rerun()
                    else:
                        st.session_state.quiz_complete = True
                        st.rerun()
                else:
                    st.warning("Please select an answer.")
            
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # Quiz Completed
            final_score = st.session_state.quiz_score
            total = len(q_data)
            percentage = (final_score / total) * 100
            
            st.balloons()
            st.markdown(f"""
            <div style="text-align: center; padding: 40px; background: rgba(30,41,59,0.5); border-radius: 16px; border: 1px solid rgba(99, 102, 241, 0.3);">
                <h1 style="font-size: 3rem; margin-bottom: 10px;">Quiz Complete! 🎉</h1>
                <h2 style="color: #6366f1;">Your Score: {final_score}/{total}</h2>
                <p style="font-size: 1.2rem; color: #94a3b8;">({percentage:.0f}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Start New Quiz"):
                st.session_state.quiz_data = []
                st.session_state.quiz_idx = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_complete = False
                st.rerun()
