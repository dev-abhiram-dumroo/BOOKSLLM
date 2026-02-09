import streamlit as st
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import tempfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="BooksLLM - Shiv Puran Search", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS with WHITE theme
st.markdown("""
    <style>
    /* Reset and Base */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        padding: 0 !important;
        max-width: 100% !important;
        background: #ffffff !important;
    }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Force white background */
    .stApp {
        background: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Navbar */
    .navbar {
        background: #3563f6;
        color: white;
        padding: 14px 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0;
    }
    
    .logo {
        font-size: 22px;
        font-weight: bold;
        color: white;
    }
    
    .nav-links {
        display: flex;
        gap: 12px;
        align-items: center;
    }
    
    .nav-links a {
        color: white;
        text-decoration: none;
        font-size: 14px;
    }
    
    .signin-btn {
        background: #ffbf47;
        color: #000;
        padding: 8px 16px;
        border-radius: 20px;
        text-decoration: none;
        font-size: 14px;
        font-weight: bold;
    }
    
    /* Hero Section */
    .hero-container {
        padding: 60px 80px;
        max-width: 1400px;
        margin: 0 auto;
        background: #ffffff;
    }
    
    .tag {
        display: inline-block;
        background: #eef2ff;
        color: #3563f6;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin-bottom: 16px;
        font-weight: 500;
    }
    
    .hero-title {
        font-size: 48px;
        margin-bottom: 10px;
        font-weight: bold;
        color: #111111;
    }
    
    .hero-title span {
        color: #3563f6;
    }
    
    .hero-description {
        margin-top: 15px;
        font-size: 16px;
        line-height: 1.6;
        color: #444444;
        max-width: 600px;
    }
    
    /* Search Box */
    .stTextInput > div > div > input {
        padding: 14px;
        font-size: 16px;
        border-radius: 6px;
        border: 1px solid #cccccc;
        background: #ffffff !important;
        color: #111111 !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999999;
    }
    
    .stButton > button {
        padding: 14px 20px;
        background: #3563f6 !important;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-size: 16px;
        width: 100%;
        margin-top: 10px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: #2952d9 !important;
    }
    
    /* Results heading */
    h3 {
        color: #111111 !important;
        font-weight: 600;
    }
    
    /* Result Container */
    .result-container {
        margin-top: 30px;
        background: #f7f9fc;
        padding: 25px;
        border-radius: 10px;
        border-left: 4px solid #3563f6;
    }
    
    .similarity-badge {
        background: #3563f6;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 15px;
        font-weight: 500;
    }
    
    .chapter-badge {
        background: #4ECDC4;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 15px;
        margin-left: 10px;
        font-weight: 500;
    }
    
    .result-container strong {
        color: #111111;
    }
    
    .sanskrit-text {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        color: #333333;
        border: 1px solid #e0e0e0;
    }
    
    .english-text {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        line-height: 1.6;
        color: #111111;
        border: 1px solid #e0e0e0;
    }
    
    /* Text color fixes */
    p, div, span {
        color: #111111;
    }
    
    /* Warning and error messages */
    .stAlert {
        background: white !important;
        color: #111111 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white !important;
        color: #111111 !important;
    }
    
    .streamlit-expanderContent {
        background: white !important;
        color: #111111 !important;
    }
    
    /* Code blocks in expander */
    code {
        background: #f5f5f5 !important;
        color: #111111 !important;
    }
    
    pre {
        background: #f5f5f5 !important;
        color: #111111 !important;
    }
    
    @media(max-width: 900px) {
        .hero-container {
            padding: 40px 20px;
        }
        
        .hero-title {
            font-size: 36px;
        }
        
        .navbar {
            padding: 14px 20px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None

# Initialize Clients
@st.cache_resource
def init_supabase():
    """Initialize Supabase client"""
    supabase: Client = create_client(
        st.secrets["SUPABASE_URL"], 
        st.secrets["SUPABASE_KEY"]
    )
    return supabase

@st.cache_resource
def load_sentence_transformer():
    """Load sentence transformer model"""
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error generating audio: {str(e)}")
        return None

supabase = init_supabase()
sentence_model = load_sentence_transformer()

# --- NAVBAR ---
st.markdown("""
    <div class="navbar">
        <div class="logo">BooksLLM</div>
        <div class="nav-links">
            <a href="#">About</a>
            <a href="#">Features</a>
            <a href="#">Docs</a>
            <a href="#">Pricing</a>
            <a class="signin-btn" href="#">Sign In</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("<div class='hero-container'>", unsafe_allow_html=True)

# Tag and Title
st.markdown("<div class='tag'>✨ Powered by AI & Embeddings</div>", unsafe_allow_html=True)
st.markdown("<h1 class='hero-title'>Books<span>LLM</span></h1>", unsafe_allow_html=True)
st.markdown("""
    <p class='hero-description'>
        Ask questions from Sanskrit books like Shiv Puran and get
        accurate English translations using embeddings and LLM-powered
        semantic search.
    </p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Search Interface
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("", placeholder="Ask your question from the book...", label_visibility="collapsed", key="search_input")

with col2:
    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                # Generate embedding
                query_embedding = sentence_model.encode(query).tolist()
                
                # Search in Supabase
                try:
                    result = supabase.rpc("match_shiv_puran_chunks", {
                        "query_embedding": query_embedding,
                        "match_threshold": 0.5,
                        "match_count": 3
                    }).execute()
                    
                    if result.data and len(result.data) > 0:
                        st.session_state.search_results = result.data
                    else:
                        st.session_state.search_results = None
                        st.warning("⚠️ No relevant verses found. Try rephrasing your question.")
                        
                except Exception as e:
                    st.session_state.search_results = None
                    st.error(f"❌ Error during search: {str(e)}")
        else:
            st.warning("Please enter a question")

# Display results
if st.session_state.search_results is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### 📜 Found {len(st.session_state.search_results)} relevant verse(s)")
    
    for idx, match in enumerate(st.session_state.search_results, 1):
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # Header with badges
        if 'similarity' in match:
            st.markdown(f"<div class='similarity-badge'>Match: {match['similarity']:.1%}</div>", unsafe_allow_html=True)
        if match.get('chapter_name'):
            st.markdown(f"<div class='chapter-badge'>{match.get('chapter_name')}</div>", unsafe_allow_html=True)
        
        st.markdown(f"**Chunk #{match.get('chunk_id', 'N/A')}**")
        st.markdown("---")
        
        # Sanskrit Text
        if match.get('content'):
            st.markdown("**📜 Sanskrit**")
            st.markdown(f"<div class='sanskrit-text'>{match.get('content')}</div>", unsafe_allow_html=True)
        
        # English Translation
        if match.get('english_translation'):
            st.markdown("**🇬🇧 English Translation**")
            english_text = match.get('english_translation', '')
            st.markdown(f"<div class='english-text'>{english_text}</div>", unsafe_allow_html=True)
            
            # Audio button
            if st.button(f"🔊 Play Audio", key=f"audio_{idx}"):
                with st.spinner("Generating audio..."):
                    audio_file = text_to_speech(english_text)
                    if audio_file:
                        audio_bytes = open(audio_file, 'rb').read()
                        st.audio(audio_bytes, format='audio/mp3')
                        os.unlink(audio_file)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
with st.expander("ℹ️ Setup Instructions"):
    st.markdown("""
    ### First-time Setup
    
    1. **Create the search function in Supabase**
    2. **Add secrets to Streamlit**
    3. **Install dependencies**: `pip install streamlit supabase sentence-transformers gtts`
    4. **Run**: `streamlit run app.py`
    """)