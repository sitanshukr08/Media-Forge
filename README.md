# Media Forge

**Media Forge** is a professional-grade media ingestion and asset processing tool built for content creators, archivists, and researchers. It provides a unified dashboard for downloading high-fidelity video, extracting studio-quality audio, and managing media assets from various social platforms directly to your local workspace.

## 🚀 Key Features

*   **Universal Ingestion**: Seamlessly processes links from **YouTube** (Videos & Shorts) and **Instagram** (Reels & Posts).
*   **High-Fidelity Video**: Supports rendering up to **4K/8K resolution** with selectable container formats (MP4, MKV).
*   **Audio Extraction**: Built-in converter for separating audio tracks into **MP3, WAV, or FLAC** formats with bitrate control (up to 320kbps).
*   **Smart Workspace**: Features a native OS folder picker to manage your output directory dynamically. Files are saved directly to disk, bypassing browser limits.
*   **Metadata Analysis**: Automatically scans video descriptions to extract and list external resource links (socials, references, etc.).
*   **Asset Grabber**: One-click extraction of high-resolution thumbnails/cover art as standardized JPG files.

## 🛠️ Installation

### Prerequisites
1.  **Python 3.8+** installed on your system.
2.  **FFmpeg** installed and added to your system PATH (Required for audio conversion and video merging).

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/sitanshukr08/Media-Forge.git
    cd Media-Forge
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

1.  Launch the application interface:
    ```bash
    streamlit run media_forge.py
    ```

2.  The interface will open in your default web browser.
3.  **Configure Workspace**: Use the sidebar to select your desired output folder on your computer.
4.  **Analyze**: Paste a link (YouTube or Instagram) into the input field.
5.  **Process**: Use the tabs to choose your desired operation (Video Render, Audio Extract, or Thumbnail Save).

## 📦 Dependencies

*   `streamlit`: For the interactive dashboard interface.
*   `yt-dlp`: Core engine for media extraction and downloading.
*   `requests`: For network handling and asset retrieval.
*   `ffmpeg`: (System Level) For media encoding and format conversion.

## ⚖️ Legal Disclaimer

This tool is designed for **educational purposes and personal archiving only**. Users are responsible for complying with the Terms of Service of the source platforms (YouTube, Instagram, etc.) and respecting copyright laws. The developers of Media Forge do not endorse copyright infringement.

---

**Author**: [sitanshukr08](https://github.com/sitanshukr08)
