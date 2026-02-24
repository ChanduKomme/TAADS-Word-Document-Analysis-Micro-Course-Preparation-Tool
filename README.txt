# PDF/Word Document Analysis & Micro Course Preparation Tool

A document analysis application that processes Word documents (.docx) and PDFs to extract text, figures, tables, and chart data for micro course preparation. Built with Streamlit for an interactive web interface.

## Features

### Document Processing
- **Text Extraction**: Extracts all text elements from Word documents and PDFs, including text within tables
- **Figure Extraction**: Extracts chart and figure images with automatic filtering by size
- **Table Extraction**: Extracts structured table data with cell coordinates
- **Chart Data Analysis**: Uses OpenCV-based computer vision to extract numerical data from charts

### Content Organization
- **Section Detection**: Automatically detects document structure using formatting-based analysis (headings, callouts, font sizes)
- **Reading Order Preservation**: Maintains original document sequence
- **Chunk Splitting**: Splits content into slide-sized chunks suitable for PowerPoint presentations
- **Figure-Section Linking**: Shows which figures are referenced in each section

### AI-Powered Features
- **Text Summarization**: Generates concise summaries for long sections using Ollama LLM
- **Learning Questions**: Generates multiple-choice and free-text review questions
- **Translation Support**: Translates content to multiple languages

### User Interface
- **Two-View Interface**: 
  - Figures View: Browse all extracted figures with JSON data download
  - Sections View: View and edit document sections in reading order
- **Drag-and-Drop Reordering**: Rearrange sections using drag-and-drop
- **Inline Text Editing**: Edit section content directly in the interface
- **Bidirectional Navigation**: Click figure references to jump to figures, navigate back to sections

### Export Options
- **PowerPoint (PPTX)**: Generate professional presentations with figures and summaries
- **HTML Slides**: Export as downloadable HTML presentation
- **JSON Data**: Download extracted chart data in JSON format

---

## How to Run on Your Computer

### Step 1: Prerequisites

Make sure you have installed:
- **Python 3.10 or higher** - Download from [python.org](https://www.python.org/downloads/)
- **Git** - Download from [git-scm.com](https://git-scm.com/downloads)

### Step 2: Clone the Repository

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

Copy and run this command to install all required packages:

```bash
pip install streamlit numpy opencv-python opencv-python-headless pandas pdfplumber pillow pymupdf python-docx python-pptx requests streamlit-sortables
```

### Step 5: Create Required Folders

```bash
# On Windows
mkdir data
mkdir data\uploads
mkdir .streamlit

# On Mac/Linux
mkdir -p data/uploads
mkdir -p .streamlit
```

### Step 6: Create Streamlit Configuration

Create a file named `config.toml` inside the `.streamlit` folder with this content:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000

[browser]
gatherUsageStats = false
```

### Step 7: Run the Application

```bash
streamlit run app.py --server.port 5000
```

### Step 8: Open in Browser

Open your web browser and go to: **http://localhost:5000**

---

## Optional: AI Summarization Setup

To enable AI-powered summarization features, install Ollama:

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and run it
3. Pull the model: `ollama pull qwen2.5`
4. Start Ollama server: `ollama serve`

The app will automatically detect Ollama running on your computer.

---

## Quick Start Guide

1. **Upload Document**: Click "Browse files" and select a Word (.docx) or PDF file
2. **Wait for Processing**: The app will extract text, figures, and tables automatically
3. **Switch Views**: Use "Figures" or "Sections" tabs to navigate content
4. **Edit Content**: Click on any section to expand and edit its text
5. **Reorder**: Drag and drop sections to change their order
6. **Export**: Click "Download PPTX" to get a PowerPoint presentation

---

## Project Structure

```
├── app.py                 # Main Streamlit application
├── graphic.py             # Graphics/chart processing utilities
├── pipeline/              # Document processing modules
│   ├── extract_text.py    # Text extraction with OCR fallback
│   ├── extract_figures.py # Figure/image extraction
│   ├── extract_tables.py  # Table extraction
│   ├── chunk_to_sections.py # Section detection and parsing
│   └── ai_summarizer.py   # AI summarization integration
├── .streamlit/            # Streamlit configuration
│   └── config.toml        # Server settings
└── data/                  # Upload and output directories
    └── uploads/           # Uploaded files stored here
```

---

## Troubleshooting

### "Module not found" error
Run the install command again:
```bash
pip install streamlit numpy opencv-python opencv-python-headless pandas pdfplumber pillow pymupdf python-docx python-pptx requests streamlit-sortables
```

### "Address already in use" error
Change the port number:
```bash
streamlit run app.py --server.port 8501
```

### App loads but shows errors
Make sure the `data/uploads` folder exists and is writable.

---

## Technical Details

- **Framework**: Streamlit
- **PDF Processing**: PyMuPDF (fitz), pdfplumber
- **Word Processing**: python-docx, zipfile
- **Computer Vision**: OpenCV for chart data extraction
- **AI Integration**: Ollama with qwen2.5 model (optional)
- **Database**: SQLite (local storage)

---

## Team

This project was developed as part of a university course on document processing and micro course preparation.

## License

This project is for educational purposes.
