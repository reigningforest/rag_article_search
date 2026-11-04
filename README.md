# Agentic RAG: Web-Based Retrieval-Augmented Generation System

## Project Overview

This project provides a web-based interface for intelligent research paper search and analysis. The system combines advanced RAG techniques with Google Gemini AI to deliver accurate, contextual answers about ArXiv research papers.

## System Architecture

The agentic RAG system follows these key steps:

1. **Query Classification**: Determines if query requires arXiv paper retrieval using Gemini
2. **Query Rewriting**: Generates optimized query variation for better retrieval using Gemini
3. **Document Retrieval**: Semantic search using rewritten query with Pinecone vector database
4. **Response Generation**: Final answer generation using Gemini with retrieved abstracts
5. **Abstract Simplification**: AI-powered simplification of complex abstracts for easier comprehension

The system provides real-time progress tracking and outputs both the main response and simplified source documents.

## Project Structure

```
rag_article_search/
├── main_app.py                   # Web interface (Streamlit)
├── main_cli.py                   # Command-line interface
├── src/                          # Core system modules
│   ├── rag/                      # RAG workflow components
│   │   ├── state.py              # RAG state definition
│   │   ├── rag_graph.py          # Graph workflow builder
│   │   └── nodes/                # Individual workflow nodes
│   │       ├── classification.py
│   │       ├── query_processing.py
│   │       ├── retrieval.py
│   │       ├── response_generation.py
│   │       └── simplify.py
│   ├── models/                   # Model management
│   │   └── model_loader.py       # Gemini, embeddings setup
│   ├── connections/              # External service connections
│   │   ├── pinecone_db.py        # Vector database operations
│   │   └── gemini_query.py       # Google Gemini integration
│   └── loaders/                  # Data processing utilities
│       ├── data_loader.py        # Dataset management
│       └── embedding.py          # Embedding operations
├── scripts/                      # Background processing
│   └── get_data_pipeline.py      # Data download and processing
├── config/                       # Configuration
│   └── config.yaml               # System settings
├── data/                         # Processed data storage
└── output/                       # Results and visualizations
```ted Generation System

An intelligent web application for searching and analyzing ArXiv research papers using advanced Retrieval-Augmented Generation (RAG) with Google Gemini. Features a user-friendly Streamlit interface with smart data management and query processing.

## Quick Start

1. **Install dependencies:**
   ```bash
   poetry install  # or pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Create .env file with your API keys
   GEMINI_API_KEY=your_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Set up Kaggle authentication:**
   - Download `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings) → API
   - Place at: `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\{username}\.kaggle\kaggle.json` (Windows)

3. **Launch the web application:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **First time setup:** Use the web interface to download and process ArXiv data
5. **Start querying:** Ask questions about research papers!

## Project Overview

This project provides a web-based interface for intelligent research paper search and analysis. The system combines advanced RAG techniques with Google Gemini AI to deliver accurate, contextual answers about ArXiv research papers.

## Features

- **🌐 Web Interface**: Clean Streamlit app for searching research papers
- **💻 Multiple Entry Points**: Web UI, CLI, and programmatic access
- **🤖 Agentic RAG**: Intelligent query processing and response generation
- **☁️ Google Gemini Integration**: Powered by Google Gemini 2.0 Flash
- **🧠 Fine-tuned Model**: SmolLM2 model for enhanced abstract simplification (GPU accelerated)
- **🗃️ Vector Database**: Pinecone integration for semantic search
- **📄 Simplified Abstracts**: AI-powered simplification of complex research papers
- **⏱️ Real-time Progress**: Live progress tracking during processing

## Installation

```bash
# Clone repository
git clone <repository-url>
cd rag_article_search

# Install dependencies
poetry install
# OR
pip install -r requirements.txt
```

**GPU Support**: CUDA-compatible GPU recommended for optimal performance with fine-tuned model

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

**Get API Keys:**
- **Gemini**: [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Pinecone**: [Pinecone Console](https://app.pinecone.io/)

**Kaggle Setup** (for data download):
1. Download `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings) → API
2. Place at: `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\{username}\.kaggle\kaggle.json` (Windows)

## Usage

### Web Interface (Recommended)
```bash
streamlit run main_app.py
```

### Command Line Interface
```bash
python main_cli.py "Your research question here"
```

### Programmatic Usage
```python
from main_app import main
result = main("Your research question here")
```

### First Time Setup
- Click "Run Data Pipeline" in the web interface to download and process ArXiv data
- Wait for processing to complete (may take several hours)
- Then start asking questions about research papers!
- **Model Selection**: Configure `use_finetuned: true/false` in `config/config.yaml` to switch between fine-tuned and Gemini models

The system shows real-time progress and outputs both the main response and simplified source documents.