import streamlit as st
import yt_dlp
import os
import re
import requests
import tkinter as tk
from tkinter import filedialog
from datetime import timedelta
from pathlib import Path

# Application Configuration
st.set_page_config(
    page_title="MediaForge Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Stylesheet
st.markdown("""
    <style>
    .stApp {
        background-color: #0f1116;
        color: #e5e7eb;
    }
    .stTextInput > div > div > input {
        background-color: #1f2937;
        color: white;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 12px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #60a5fa;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1f2937;
        border-radius: 8px;
        color: #d1d5db;
        border: 1px solid #374151;
        padding: 0 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        height: 3.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .status-panel {
        background-color: #1f2937;
        border-left: 4px solid #10b981;
        padding: 16px;
        border-radius: 6px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# State Management
if 'work_dir' not in st.session_state:
    st.session_state.work_dir = str(Path.home() / "Downloads")
if 'media_data' not in st.session_state:
    st.session_state.media_data = None
if 'active_url' not in st.session_state:
    st.session_state.active_url = ""

# Utilities
def select_directory():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(master=root)
    root.destroy()
    return folder

def extract_links(text):
    if not text: return []
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, text)

def save_thumbnail(url, title, folder):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            clean_title = "".join([c for c in title if c.isalnum() or c in (' ','-','_')]).rstrip()
            save_path = os.path.join(folder, f"{clean_title}_thumb.jpg")
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return save_path
    except Exception:
        return None

def progress_callback(d):
    if d['status'] == 'downloading':
        try:
            p = d.get('_percent_str', '0%').replace('%', '')
            val = float(p) / 100
            progress_bar.progress(min(val, 1.0))
            status_placeholder.markdown(
                f"""
                <div style="font-family: monospace; color: #d1d5db; font-size: 0.9em; display: flex; justify-content: space-between;">
                    <span><b>STATUS:</b> DOWNLOADING</span>
                    <span><b>PROGRESS:</b> {d.get('_percent_str')}</span>
                    <span><b>SPEED:</b> {d.get('_speed_str')}</span>
                    <span><b>ETA:</b> {d.get('_eta_str')}</span>
                </div>
                """, unsafe_allow_html=True
            )
        except:
            pass
    elif d['status'] == 'finished':
        progress_bar.progress(1.0)
        status_placeholder.markdown("⚙️ **Finalizing container format...**")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Workspace")
    st.caption("Destination Directory")
    st.code(st.session_state.work_dir)
    
    if st.button("📂 Change Directory"):
        new_path = select_directory()
        if new_path:
            st.session_state.work_dir = new_path
            st.rerun()

    st.markdown("---")
    st.caption("Version 2.3 | Supported Platforms: "
    "Instagram, Youtube, Shorts")

# Main Interface
col_logo, col_header = st.columns([1, 10])
with col_header:
    st.title("MediaForge Studio")
    st.markdown("Universal Media Ingestion & Processing")

# Input Section
url_in = st.text_input("Media Source URL", placeholder="Paste YouTube, Shorts, or Instagram Reel link here...", label_visibility="collapsed")

if url_in:
    if st.session_state.active_url != url_in:
        st.session_state.active_url = url_in
        st.session_state.media_data = None
        
        with st.spinner("🔄 Analying manifest..."):
            try:
                opts = {'quiet': True, 'extract_flat': False, 'no_warnings': True}
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url_in, download=False)
                    st.session_state.media_data = info
            except Exception as e:
                st.error(f"Analysis Failed: {str(e)}")

# Dashboard
if st.session_state.media_data:
    data = st.session_state.media_data
    
    st.markdown("---")
    
    # Asset Overview
    col_preview, col_meta = st.columns([1.2, 2])
    with col_preview:
        st.image(data.get('thumbnail'), use_container_width=True)
    
    with col_meta:
        st.subheader(data.get('title', 'Untitled Asset'))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Publisher", data.get('uploader', 'Unknown'))
        m2.metric("Duration", str(timedelta(seconds=data.get('duration', 0))))
        m3.metric("Views", f"{data.get('view_count', 0):,}")
        
        # Link Extraction Module
        description = data.get('description', '')
        links = extract_links(description)
        
        with st.expander(f"🔗 Metadata Links ({len(links)})", expanded=False):
            if links:
                for link in links:
                    st.markdown(f"- [{link}]({link})")
            else:
                st.caption("No embedded links detected.")

    # Processing Tabs
    st.markdown("### 🛠️ Production Tools")
    tab_video, tab_audio, tab_assets = st.tabs(["Video Render", "Audio Extraction", "Asset Library"])

    # Video Tab
    with tab_video:
        c1, c2 = st.columns(2)
        with c1:
            formats = data.get('formats', [])
            res_set = {f['height'] for f in formats if f.get('height')}
            sorted_res = sorted(list(res_set), reverse=True)
            res_choice = st.selectbox("Resolution", sorted_res, format_func=lambda x: f"{x}p")
            
        with c2:
            container = st.selectbox("Container", ["mp4", "mkv"])
            
        if st.button("🚀 Render Output", key="btn_vid"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            ydl_opts = {
                'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s',
                'format': f'bestvideo[height<={res_choice}]+bestaudio/best[height<={res_choice}]',
                'merge_output_format': container,
                'progress_hooks': [progress_callback],
                'writethumbnail': True,
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url_in])
                
                st.markdown(f'<div class="status-panel">✅ <b>Render Complete</b><br>File saved to workspace.</div>', unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"Render Error: {e}")

    # Audio Tab
    with tab_audio:
        c1, c2 = st.columns(2)
        with c1:
            aud_fmt = st.selectbox("Format", ["mp3", "wav", "flac"])
        with c2:
            aud_quality = st.selectbox("Bitrate", ["320 kbps (Studio)", "192 kbps (High)", "128 kbps (Standard)"])
            bitrate_val = aud_quality.split()[0]

        if st.button("🎵 Extract Audio", key="btn_aud"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            ydl_opts = {
                'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s',
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': aud_fmt,
                    'preferredquality': bitrate_val,
                }],
                'progress_hooks': [progress_callback],
            }
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url_in])
                st.markdown(f'<div class="status-panel">✅ <b>Extraction Complete</b><br>Audio file saved to workspace.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Extraction Error: {e}")

    # Assets Tab
    with tab_assets:
        col_t1, col_t2 = st.columns([1, 4])
        with col_t1:
            st.image(data.get('thumbnail'), width=180)
        with col_t2:
            st.subheader("High-Fidelity Thumbnail")
            st.write("Export cover art as localized JPG resource.")
            if st.button("💾 Save Asset", key="btn_thumb"):
                path = save_thumbnail(data.get('thumbnail'), data.get('title'), st.session_state.work_dir)
                if path:
                    st.success(f"Asset written to disk: {os.path.basename(path)}")
                else:
                    st.error("Write permission denied or network error.")