import streamlit as st
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import tempfile
import os
import html  # Import for HTML escaping

# --- CONFIGURATION ---
st.set_page_config(page_title="Shiv Puran Search", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for clean UI
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 15px;
    }
    .search-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .result-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 25px;
        margin-top: 30px;
        border-left: 4px solid #FF6B35;
    }
    .similarity-badge {
        background-color: #FF6B35;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 15px;
    }
    .chapter-badge {
        background-color: #4ECDC4;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 14px;
        display: inline-block;
        margin-bottom: 15px;
        margin-left: 10px;
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
    """Load sentence transformer model - MUST match the model used for embeddings"""
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

# --- MAIN APP UI ---
st.markdown("<div class='search-container'>", unsafe_allow_html=True)

st.title("🔱 Shiv Puran Search")
st.markdown("*Ask questions about the Shiv Puran and get relevant verses*")
st.markdown("###")

# Search Interface
query = st.text_input("", placeholder="Ask your question about Shiv Puran...", label_visibility="collapsed")

# Number of results slider
num_results = st.slider("Number of results", min_value=1, max_value=5, value=1, help="How many relevant chunks to retrieve")

if st.button("🔍 Search", type="primary", use_container_width=True) and query:
    with st.spinner("Searching..."):
        # Generate embedding using Sentence Transformers
        query_embedding = sentence_model.encode(query).tolist()
        
        # Search in Supabase
        try:
            result = supabase.rpc("match_shiv_puran_chunks", {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": num_results
            }).execute()
            
            if result.data and len(result.data) > 0:
                st.session_state.search_results = result.data
            else:
                st.session_state.search_results = None
                st.warning("⚠️ No relevant verses found. Try rephrasing your question.")
                
        except Exception as e:
            st.session_state.search_results = None
            st.error(f"❌ Error during search: {str(e)}")
            st.info("💡 Make sure you've created the search function in Supabase (see setup instructions)")

# Display results if they exist in session state
if st.session_state.search_results is not None:
    st.markdown(f"### Found {len(st.session_state.search_results)} relevant verse(s)")
    
    for idx, match in enumerate(st.session_state.search_results, 1):
        # Display result
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        
        # Header with similarity score and chapter
        col1, col2 = st.columns([3, 1])
        with col1:
            if 'similarity' in match:
                # Safe: Using f-string with number, no HTML from database
                st.markdown(f"<div class='similarity-badge'>Match: {match['similarity']:.1%}</div>", unsafe_allow_html=True)
            
            if match.get('chapter_name'):
                # ✅ SECURE: Escape HTML before rendering
                safe_chapter = html.escape(match.get('chapter_name', ''))
                st.markdown(f"<div class='chapter-badge'>{safe_chapter}</div>", unsafe_allow_html=True)
        
        # ✅ SECURE: Using st.text() instead of st.markdown for database content
        chunk_id = match.get('chunk_id', match.get('id', 'N/A'))
        st.markdown(f"**Chunk #{chunk_id}**")
        st.markdown("---")
        
        # Sanskrit Text - SECURE: st.text doesn't render HTML
        if match.get('content'):
            st.markdown("**📜 Sanskrit**")
            st.text(match.get('content'))  # ✅ SAFE: st.text() auto-escapes
            st.markdown("")
        
        # English Translation - SECURE: st.write doesn't render HTML
        if match.get('english_translation'):
            st.markdown("**🇬🇧 English Translation**")
            english_text = match.get('english_translation', '')
            st.write(english_text)  # ✅ SAFE: st.write() auto-escapes
            
            # Audio button for each result
            if st.button(f"🔊 Play Audio", key=f"audio_{idx}", use_container_width=True):
                with st.spinner("Generating audio..."):
                    audio_file = text_to_speech(english_text)
                    if audio_file:
                        audio_bytes = open(audio_file, 'rb').read()
                        st.audio(audio_bytes, format='audio/mp3')
                        os.unlink(audio_file)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

st.markdown("</div>", unsafe_allow_html=True)

# Footer with instructions
st.markdown("---")
with st.expander("ℹ️ Setup Instructions"):
    st.markdown("""
    ### First-time Setup
    
    1. **Create the search function in Supabase** (run this SQL once):
    
    ```sql
    CREATE OR REPLACE FUNCTION match_shiv_puran_chunks(
        query_embedding vector(768),
        match_threshold float DEFAULT 0.5,
        match_count int DEFAULT 3
    )
    RETURNS TABLE (
        chunk_id int,
        content text,
        english_translation text,
        chapter_name text,
        similarity float
    )
    LANGUAGE sql STABLE
    AS $$
        SELECT
            id as chunk_id,
            content,
            English_translation as english_translation,
            chapter_name,
            1 - (embedding <=> query_embedding) as similarity
        FROM shiv_puran_chunks
        WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> query_embedding) > match_threshold
        ORDER BY embedding <=> query_embedding
        LIMIT match_count;
    $$;
    ```
    
    2. **Add secrets to Streamlit** (in `.streamlit/secrets.toml`):
    
    ```toml
    SUPABASE_URL = "your-supabase-url"
    SUPABASE_KEY = "your-supabase-key"
    ```
    
    3. **Install dependencies**:
    ```bash
    pip install streamlit supabase sentence-transformers gtts
    ```
    
    4. **Run the app**:
    ```bash
    streamlit run shiv_puran_search_app.py
    ```
    """)
