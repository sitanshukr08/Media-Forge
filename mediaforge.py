"""
MediaForge v8.0
Professional Media Ingestion & Analysis Suite.

This tool allows users to:
1. Download high-fidelity video/audio assets from YouTube/Instagram.
2. Transcribe speech-to-text using OpenAI Whisper.
3. Analyze content semantics (Summarization, Topic Classification, Keyword Extraction).
4. Automatically segment videos into semantic chapters using embedding analysis.
"""

import streamlit as st
import yt_dlp
import os
import uuid
import time
import json
import torch
import whisper
import yake
import numpy as np
from datetime import timedelta
from pathlib import Path

# Optional imports for AI features (Lazy loaded to ensure basic functionality works without them)
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    pass

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="MediaForge | Media Architect",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0d11; color: #c9d1d9; }
    .stButton>button { background-color: #238636; border: none; height: 3rem; font-weight: 600; color: white; border-radius: 6px; }
    .stButton>button:hover { background-color: #2ea043; }
    .metric-card { background: #161b22; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; }
    .chapter-card { background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 8px; }
    .timestamp-badge { background: #238636; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-family: monospace; }
    .tag-pill { background-color: #1f6feb; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.85em; margin-right: 5px; display: inline-block; margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'work_dir' not in st.session_state: st.session_state.work_dir = str(Path.home() / "Downloads")
if 'ai_results' not in st.session_state: st.session_state.ai_results = {}
if 'media_info' not in st.session_state: st.session_state.media_info = None
if 'active_url' not in st.session_state: st.session_state.active_url = ""

# --- HELPER UTILITIES ---
# Removed select_directory() function as it depended on tkinter

def format_timestamp(seconds):
    """Formats seconds into HH:MM:SS,mmm string for SRT."""
    millis = int(round((seconds % 1) * 1000))
    td = timedelta(seconds=int(seconds))
    return f"{str(td).zfill(8)},{millis:03d}"

def generate_srt(segments):
    """Converts Whisper segments into SRT subtitle format."""
    srt = ""
    for i, enumerate_seg in enumerate(segments):
        start = format_timestamp(enumerate_seg['start'])
        end = format_timestamp(enumerate_seg['end'])
        srt += f"{i+1}\n{start} --> {end}\n{enumerate_seg['text'].strip()}\n\n"
    return srt

def download_audio_temp(url):
    """Downloads temporary low-bitrate audio for AI analysis."""
    uid = uuid.uuid4().hex
    path = f"temp_ai_audio_{uid}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': path,
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3','preferredquality': '32'}],
        'quiet': True, 'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    return path + ".mp3"

# --- AI ENGINE ---
class AIEngine:
    """Core class managing ML models and inference logic."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_cache = {}
        self.embedder = None
        self.summarizer = None
        self.classifier = None
        # Map generic language codes to YAKE compatible ones
        self.YAKE_LANG_MAP = { "zh": "zh-cn", "no": "nb", "jw": "id" }

    def load_whisper(self, size="base"):
        """Loads and caches Whisper ASR model."""
        if size not in self.whisper_cache:
            self.whisper_cache[size] = whisper.load_model(size, device=self.device)
        return self.whisper_cache[size]

    def load_embedder(self):
        """Loads Sentence Transformer for semantic analysis."""
        try:
            from sentence_transformers import SentenceTransformer
            if self.embedder is None:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            return self.embedder
        except ImportError:
            st.error("Missing library: sentence-transformers. Feature disabled.")
            return None

    def load_summarizer(self):
        """Loads DistilBART summarization pipeline."""
        if self.summarizer is None:
            dev_id = 0 if self.device == "cuda" else -1
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=dev_id)
        return self.summarizer

    def load_classifier(self):
        """Loads Zero-Shot Classifier for topic detection."""
        if self.classifier is None:
            dev_id = 0 if self.device == "cuda" else -1
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=dev_id)
        return self.classifier

    def transcribe_audio(self, audio_path, task="transcribe", model_size="base"):
        """Runs ASR inference with anti-hallucination thresholds."""
        model = self.load_whisper(model_size)
        return model.transcribe(
            audio_path, 
            task=task, 
            condition_on_previous_text=False, # Prevents timestamp drift
            fp16=(self.device=="cuda"), 
            compression_ratio_threshold=2.4, 
            logprob_threshold=-1.0
        )

    def detect_chapters(self, segments):
        """
        Detects semantic chapters by embedding text blocks and finding similarity dips.
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return []

        embedder = self.load_embedder()
        if not embedder: return []
        
        # 1. Group segments into 30s+ blocks to stabilize embeddings
        blocks = []
        current_block = {"text": "", "start": segments[0]['start'], "end": 0}
        
        for seg in segments:
            current_block["text"] += " " + seg["text"]
            current_block["end"] = seg["end"]
            
            # Create block if duration > 30s or length > 500 chars
            if (seg["end"] - current_block["start"] > 30) or len(current_block["text"]) > 500:
                blocks.append(current_block)
                current_block = {"text": "", "start": seg["end"], "end": 0}
        
        if current_block["text"]: blocks.append(current_block)
        if len(blocks) < 2: return []

        # 2. Compute Embeddings and Similarity
        texts = [b["text"] for b in blocks]
        embeddings = embedder.encode(texts)
        
        sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]
        
        # 3. Detect Boundaries (Dynamic Threshold)
        threshold = np.mean(sims) - 0.5 * np.std(sims)
        chapters = []
        current_chap_start_idx = 0

        for i in range(len(sims)):
            current_duration = blocks[i]['end'] - blocks[current_chap_start_idx]['start']
            
            # Trigger new chapter if similarity is low AND duration is sufficient
            if sims[i] < threshold and current_duration > 60:
                chap_text = " ".join([b['text'] for b in blocks[current_chap_start_idx:i+1]])
                title = self.generate_smart_title(chap_text)
                
                chapters.append({
                    "start": blocks[current_chap_start_idx]['start'],
                    "end": blocks[i]['end'],
                    "title": title
                })
                current_chap_start_idx = i + 1

        # Add Final Chapter
        final_text = " ".join([b['text'] for b in blocks[current_chap_start_idx:]])
        chapters.append({
            "start": blocks[current_chap_start_idx]['start'],
            "end": blocks[-1]['end'],
            "title": self.generate_smart_title(final_text)
        })
            
        return chapters

    def generate_smart_title(self, text):
        """Generates a concise title for a text block using summarization."""
        summ_pipe = self.load_summarizer()
        try:
            res = summ_pipe(text[:1500], max_length=15, min_length=5, do_sample=False)
            return res[0]['summary_text'].strip().title()
        except:
            return "Untitled Segment"

    def classify_topic(self, text):
        """Classifies content into broad categories."""
        classifier = self.load_classifier()
        labels = ["Technology", "Politics", "Entertainment", "Education", "Finance", "Gaming", "Health", "Science"]
        res = classifier(text[:1024], labels)
        return res['labels'][0], res['scores'][0]

    def generate_summary(self, text):
        """Map-Reduce summarization for long transcripts."""
        summ_pipe = self.load_summarizer()
        chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
        chunk_summaries = []
        for chunk in chunks:
            if len(chunk) > 100:
                try:
                    res = summ_pipe(chunk, max_length=130, min_length=30, do_sample=False)
                    chunk_summaries.append(res[0]['summary_text'])
                except RuntimeError: pass 
        return "\n\n".join(chunk_summaries)

    def extract_keywords(self, text, lang="en"):
        """Extracts keywords using YAKE."""
        target_lang = self.YAKE_LANG_MAP.get(lang, lang)
        kw = yake.KeywordExtractor(lan=target_lang, n=2, dedupLim=0.9, top=8)
        return [k[0] for k in kw.extract_keywords(text)]

@st.cache_resource
def get_engine(): return AIEngine()
engine = get_engine()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("💠 Config")
    
    st.markdown("### 📂 Output Path")
    # Replaced button with text_input to avoid Tkinter crashes in hosted environments
    new_dir = st.text_input("Directory", value=st.session_state.work_dir)
    if new_dir != st.session_state.work_dir:
        st.session_state.work_dir = new_dir
    
    st.markdown("### 🧠 AI Model Settings")
    model_size = st.selectbox("Whisper Size", ["base", "small", "medium", "large"], index=1)
    task_type = st.radio("Task", ["Transcribe", "Translate to English"])
    task_code = "translate" if "Translate" in task_type else "transcribe"
    
    st.divider()
    st.caption("MediaForge v8.0 | Professional Edition")

# --- MAIN INTERFACE ---
st.title("MediaForge | AI Media Architect")
url_in = st.text_input("Source URL", placeholder="Paste YouTube link...", label_visibility="collapsed")

# Initialize Session Data if new URL
if url_in:
    if st.session_state.active_url != url_in:
        st.session_state.media_info = None
        st.session_state.active_url = url_in
        with st.spinner("🔄 Fetching Manifest..."):
            try:
                with yt_dlp.YoutubeDL({'quiet':True}) as ydl:
                    st.session_state.media_info = ydl.extract_info(url_in, download=False)
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# DASHBOARD
if st.session_state.media_info:
    data = st.session_state.media_info
    c1, c2 = st.columns([1, 2.5])
    with c1: st.image(data.get('thumbnail'), use_container_width=True)
    with c2:
        st.subheader(data.get('title'))
        st.caption(f"{data.get('uploader')} • {str(timedelta(seconds=data.get('duration', 0)))} • {data.get('view_count', 0):,} views")

    st.markdown("---")
    
    # APP TABS
    tab_down, tab_ai, tab_export = st.tabs(["⬇️ Media Downloader", "🧠 Neural Analysis", "📦 Export Data"])

    # --- TAB 1: MEDIA DOWNLOADER ---
    with tab_down:
        st.markdown("#### Download Assets")
        c1, c2 = st.columns(2)
        
        def progress_hook(d):
            if d['status'] == 'downloading':
                try:
                    p = d.get('_percent_str', '0%').replace('%', '')
                    progress_bar.progress(min(float(p)/100, 1.0))
                except: pass

        with c1:
            formats = data.get('formats', [])
            res_set = {f['height'] for f in formats if f.get('height')}
            res_choice = st.selectbox("Video Resolution", sorted(list(res_set), reverse=True), format_func=lambda x: f"{x}p")
            
            if st.button("🎬 Download Video"):
                progress_bar = st.progress(0)
                try:
                    opts = {
                        'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s',
                        'format': f'bestvideo[height<={res_choice}]+bestaudio/best[height<={res_choice}]',
                        'progress_hooks': [progress_hook],
                        'merge_output_format': 'mp4'
                    }
                    with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url_in])
                    st.success(f"Saved to {st.session_state.work_dir}")
                except Exception as e:
                    st.error(f"Download Failed: {e}")

        with c2:
            aud_fmt = st.selectbox("Audio Format", ["mp3", "wav", "flac"])
            if st.button("🎵 Download Audio"):
                progress_bar = st.progress(0)
                try:
                    opts = {
                        'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s',
                        'format': 'bestaudio/best',
                        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': aud_fmt}],
                        'progress_hooks': [progress_hook]
                    }
                    with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([url_in])
                    st.success(f"Saved to {st.session_state.work_dir}")
                except Exception as e:
                     st.error(f"Audio Failed: {e}")

    # --- TAB 2: AI ANALYSIS ---
    with tab_ai:
        cache_key = f"{url_in}_{model_size}_{task_code}"
        
        # Check cache before running expensive inference
        if not st.session_state.ai_results.get(cache_key):
            if st.button("🚀 Run Analysis (Chapters + Summary)"):
                status = st.status("Running AI Pipeline...", expanded=True)
                try:
                    # 1. Audio Extraction
                    status.write("📡 Extracting audio stream...")
                    audio_path = download_audio_temp(url_in)
                    
                    # 2. Transcription
                    status.write(f"🎙️ Whisper ({model_size}): Transcribing...")
                    res = engine.transcribe_audio(audio_path, task=task_code, model_size=model_size)
                    full_text = res["text"]
                    lang = "en" if task_code == "translate" else res["language"]
                    
                    # 3. Chapter Detection
                    status.write("📑 Semantic Engine: Detecting Chapters...")
                    chapters = engine.detect_chapters(res["segments"])
                    
                    # 4. Summarization & Classification
                    status.write("📝 NLP: Summarizing & Classifying...")
                    summary = engine.generate_summary(full_text) if lang == "en" else "Non-English content."
                    topic, conf = engine.classify_topic(summary if lang == "en" else full_text)
                    keywords = engine.extract_keywords(full_text, lang)
                    
                    # Store Results
                    st.session_state.ai_results[cache_key] = {
                        "chapters": chapters,
                        "summary": summary,
                        "topic": topic, "conf": conf,
                        "keywords": keywords,
                        "text": full_text,
                        "srt": generate_srt(res["segments"])
                    }
                    if os.path.exists(audio_path): os.remove(audio_path)
                    status.update(label="Complete", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(str(e))
        
        # DISPLAY RESULTS
        if st.session_state.ai_results.get(cache_key):
            r = st.session_state.ai_results[cache_key]
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("📑 Smart Chapters")
                if r['chapters']:
                    for ch in r['chapters']:
                        st.markdown(f"""
                        <div class="chapter-card">
                            <span class="timestamp-badge">{format_timestamp(ch['start'])} - {format_timestamp(ch['end'])}</span>
                            <b>{ch['title']}</b>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No distinct semantic chapters found.")
            
            with c2:
                st.subheader("📊 Insights")
                st.metric("Topic", f"{r['topic']} ({r['conf']:.0%})")
                st.write("**Keywords:**")
                st.markdown("".join([f"<span class='tag-pill'>{k}</span>" for k in r['keywords']]), unsafe_allow_html=True)
                with st.expander("Summary"):
                    st.write(r['summary'])

    # --- TAB 3: EXPORTS ---
    with tab_export:
        r = st.session_state.ai_results.get(f"{url_in}_{model_size}_{task_code}")
        if r:
            st.download_button("💾 JSON Report", json.dumps(r, indent=4, default=str), "meta.json")
            st.download_button("⏱️ Subtitles (SRT)", r['srt'], "subs.srt")
            st.download_button("📜 Transcript", r['text'], "trans.txt")
        else:
            st.info("Run analysis to enable exports.")
