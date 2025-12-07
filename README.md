# 💠 MediaForge

**Professional Media Ingestion & Analysis Suite**

MediaForge is a powerful Streamlit-based application that combines media downloading capabilities with advanced AI-powered analysis. Download high-quality video and audio from YouTube, transcribe content using OpenAI Whisper, and leverage cutting-edge NLP for automatic summarization, topic classification, and semantic chapter detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup](#step-by-step-setup)
- [Usage](#-usage)
  - [Starting the Application](#starting-the-application)
  - [Downloading Media](#downloading-media)
  - [AI Analysis](#ai-analysis)
  - [Exporting Data](#exporting-data)
- [Configuration](#-configuration)
- [Dependencies](#-dependencies)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎬 Media Downloader
- **Video Download**: Download videos in multiple resolutions (144p to 4K)
- **Audio Extraction**: Extract audio in MP3, WAV, or FLAC formats
- **Progress Tracking**: Real-time download progress indicator
- **Custom Output Path**: Choose your preferred download directory

### 🧠 Neural Analysis (AI-Powered)
- **Speech-to-Text Transcription**: Powered by OpenAI Whisper with multiple model sizes
- **Translation Support**: Automatically translate non-English audio to English
- **Smart Chapter Detection**: Semantic segmentation using sentence embeddings
- **Topic Classification**: Zero-shot classification into 8 categories
- **Keyword Extraction**: Automatic keyword extraction using YAKE algorithm
- **Content Summarization**: Map-reduce summarization for long transcripts

### 📦 Export Options
- **JSON Report**: Complete analysis data in structured JSON format
- **SRT Subtitles**: Standard subtitle format with timestamps
- **Plain Text Transcript**: Full transcript as text file

---

## 🚀 Installation

### Prerequisites

Before installing MediaForge, ensure you have the following:

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8+ | Required |
| FFmpeg | Latest | Required for audio/video processing |
| CUDA (Optional) | 11.0+ | For GPU acceleration |

### Step-by-Step Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mediaforge.git
cd mediaforge
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### 3. Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install dependencies manually:

```bash
# Core dependencies
pip install streamlit yt-dlp torch whisper yake numpy

# AI/ML dependencies (for full functionality)
pip install transformers sentence-transformers scikit-learn
```

#### 5.  Verify Installation

```bash
python -c "import streamlit; import whisper; print('Installation successful!')"
```

---

## 📖 Usage

### Starting the Application

Launch MediaForge using Streamlit:

```bash
streamlit run mediaforge_app.py
```

The application will open in your default browser at `http://localhost:8501`

### Downloading Media

1. **Paste URL**: Enter a YouTube URL in the input field
2. **Wait for Metadata**: The app will fetch video information automatically
3. **Navigate to "Media Downloader" Tab**
4. **Select Options**:
   - For video: Choose resolution and click "🎬 Download Video"
   - For audio: Choose format (MP3/WAV/FLAC) and click "🎵 Download Audio"
5.  **Monitor Progress**: Watch the progress bar until completion

### AI Analysis

1. **Paste URL**: Enter a YouTube URL
2. **Configure Settings** (Sidebar):
   - Select Whisper model size (base/small/medium/large)
   - Choose task type (Transcribe or Translate)
3.  **Navigate to "Neural Analysis" Tab**
4. **Click "🚀 Run Analysis"**
5. **Wait for Processing**: The pipeline will:
   - Extract audio stream
   - Transcribe using Whisper
   - Detect semantic chapters
   - Generate summary and classify topic
   - Extract keywords
6. **View Results**: Explore chapters, insights, and summary

### Exporting Data

1. **Complete AI Analysis** first
2.  **Navigate to "Export Data" Tab**
3. **Download Options**:
   - 💾 **JSON Report**: Complete structured data
   - ⏱️ **Subtitles (SRT)**: Timestamped subtitles
   - 📜 **Transcript**: Plain text transcript

---

## ⚙️ Configuration

### Sidebar Options

| Setting | Description | Default |
|---------|-------------|---------|
| **Output Path** | Directory for downloaded files | `~/Downloads` |
| **Whisper Size** | Model size (affects accuracy vs speed) | `small` |
| **Task** | Transcribe or Translate to English | `Transcribe` |

### Whisper Model Comparison

| Model | Parameters | Speed | Accuracy | VRAM Required |
|-------|------------|-------|----------|---------------|
| `base` | 74M | Fastest | Good | ~1GB |
| `small` | 244M | Fast | Better | ~2GB |
| `medium` | 769M | Medium | Great | ~5GB |
| `large` | 1550M | Slow | Best | ~10GB |

---

## 📦 Dependencies

### Core Dependencies

```
streamlit>=1.28.0
yt-dlp>=2023.10.0
torch>=2.0.0
openai-whisper>=20231117
yake>=0.4.8
numpy>=1.24.0
```

### AI/ML Dependencies (Optional but Recommended)

```
transformers>=4. 35.0
sentence-transformers>=2.2.2
scikit-learn>=1.3. 0
```

### System Dependencies

- **FFmpeg**: Required for audio/video processing
- **Tkinter**: For native folder picker dialog (usually included with Python)

---

## 🔧 Troubleshooting

### Common Issues

#### "FFmpeg not found" Error
```bash
# Verify FFmpeg installation
ffmpeg -version

# If not found, install FFmpeg (see Installation section)
```

#### CUDA/GPU Issues
```python
# Check if PyTorch detects GPU
import torch
print(torch.cuda.is_available())  # Should return True for GPU support
```

#### Memory Errors with Large Models
- Use a smaller Whisper model (`base` or `small`)
- Close other applications to free up RAM/VRAM
- Consider processing shorter videos

#### Download Failures
- Check your internet connection
- Verify the URL is valid and accessible
- Some videos may be region-restricted or private

#### Missing AI Features
If summarization or chapter detection isn't working:
```bash
pip install transformers sentence-transformers scikit-learn
```

### Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed for 5-10x faster transcription
2. **Start Small**: Use `base` model for testing, upgrade for production
3.  **Cached Results**: Analysis results are cached - rerunning is instant

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/mediaforge.git
cd mediaforge

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests (if available)
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Media downloading
- [Streamlit](https://streamlit.io/) - Web interface
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [YAKE](https://github.com/LIAAD/yake) - Keyword extraction

---

## 📧 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [Issues](https://github.com/yourusername/mediaforge/issues)
3.  Open a new issue with detailed information

---

<p align="center">
  Made with ❤️ by Sitanshu
</p>

<p align="center">
  <b>⭐ Star this repo if you find it useful!</b>
</p>
