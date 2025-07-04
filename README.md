# Agentic RAG: A Modular Framework for Advanced Retrieval-Augmented Generation

This repository contains a modular, advanced Retrieval-Augmented Generation (RAG) system that leverages agentic components and Google Gemini to enhance the quality of responses from Large Language Models (LLMs).

## Table of Contents
- [Agentic RAG: A Modular Framework for Advanced Retrieval-Augmented Generation](#agentic-rag-a-modular-framework-for-advanced-retrieval-augmented-generation)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [System Architecture](#system-architecture)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Environment Variables](#environment-variables)
    - [Data Setup](#data-setup)
  - [Usage](#usage)
    - [Interactive Mode](#interactive-mode)
    - [Batch Processing](#batch-processing)
  - [Docker Setup](#docker-setup)
    - [Prerequisites for Docker](#prerequisites-for-docker)
    - [Production Deployment](#production-deployment)
    - [Manual Docker Build](#manual-docker-build)
    - [Development with Docker](#development-with-docker)
  - [Configuration](#configuration)
  - [Testing](#testing)
    - [Test Structure](#test-structure)
  - [Additional Features](#additional-features)
    - [RAG Comparison Framework](#rag-comparison-framework)
    - [Legacy Scripts](#legacy-scripts)
  - [API Documentation](#api-documentation)
    - [Core Components](#core-components)
    - [Key Node Functions](#key-node-functions)
    - [Workflow Routing](#workflow-routing)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Project Overview

This project implements a modular agentic RAG system designed to improve question-answering performance on academic papers. The system uses intelligent agents to classify queries, rewrite them for better retrieval, and simplify retrieved content using Google Gemini before generating final responses.

## Features

- **🤖 Agentic Components**: Query classifier, rewriter, and Gemini-powered processing
- **🏗️ Modular Architecture**: Clean separation of concerns with dedicated modules
- **🔄 LangGraph Workflow**: Visual workflow management with conditional routing
- **☁️ Google Gemini Integration**: All LLM operations powered by Google Gemini 2.0 Flash
- **🗃️ Vector Database**: Pinecone integration for semantic search with cached embeddings
- **📊 Batch Processing**: Support for processing multiple queries with CSV output
- **🐳 Docker Support**: Containerized deployment with development and production configurations
- **🧪 Comprehensive Testing**: Full test suite with pytest integration

## System Architecture

The agentic RAG system follows these steps:

1. **Query Classification**: Determines if query requires arXiv paper retrieval using Gemini
2. **Query Rewriting**: Generates multiple query variations for better retrieval using Gemini
3. **Document Retrieval**: Semantic search using concatenated queries with Pinecone vector database
4. **Abstract Simplification**: Gemini-powered simplification for better comprehension
5. **Response Generation**: Final answer generation using Gemini with enhanced context

All LLM operations (classification, rewriting, simplification, and final response) are powered by Google Gemini 2.0 Flash.

![System Workflow](./output/fine_tuned_rag_graph.png)

## Project Structure

```
rag_article_search/
├── src/                          # Main source code
│   ├── core/                     # Core RAG components
│   │   ├── __init__.py
│   │   ├── state.py              # RAG state definition
│   │   ├── nodes.py              # LangGraph node functions
│   │   └── rag_graph.py          # Graph workflow builder
│   ├── models/                   # Model management
│   │   ├── __init__.py
│   │   └── model_loader.py       # Gemini, embeddings setup
│   ├── connections/              # External service connections
│   │   ├── __init__.py
│   │   ├── pinecone_db.py        # Vector database operations
│   │   └── gemini_query.py       # Google Gemini integration
│   ├── loaders/                  # Data loading utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py        # Dataset management
│   │   └── embedding.py          # Embedding operations
│   ├── processing/               # Data processing utilities
│   │   └── __init__.py
│   └── visualization/            # Graph visualization
│       ├── __init__.py
│       └── graph_viz.py          # Workflow visualization
├── test/                         # Comprehensive test suite
│   ├── test_core/                # Core functionality tests
│   ├── test_connections/         # Connection tests
│   ├── test_loaders/             # Data loader tests
│   ├── test_models/              # Model tests
│   └── conftest.py               # Test configuration
├── utils/                        # Utility functions
│   └── logger.py                 # Logging configuration
├── scripts/                      # Additional scripts
│   └── run_pipeline.py           # Data processing pipeline
├── config/                       # Configuration files
│   └── config.yaml               # System configuration
├── main.py                       # Main application entry point
├── download_filter_embed_upsert.py  # Data preparation script
├── batch_processor.py            # Batch processing script
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Poetry configuration
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This file
```

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Kaggle Account** for arXiv dataset access
- **API Keys**: Pinecone and Google Gemini
- **Docker** (optional, for containerized deployment)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag_article_search
   ```

2. **Create virtual environment:**
   ```bash
   # Using conda
   conda create -n rag python=3.11
   conda activate rag
   
   # Or using venv
   python -m venv rag
   source rag/bin/activate  # Windows: rag\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Using pip
   pip install -r requirements.txt
   
   # Or using Poetry (recommended)
   poetry install
   ```

### Environment Setup

Set up your environment variables as described in the [Environment Variables](#environment-variables) section.

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

You can also set these as system environment variables or pass them directly to Docker containers.

### Data Setup

1. **Download and process data:**
   ```bash
   # Using the main data processing script
   python download_filter_embed_upsert.py
   
   # Or using the modular script (recommended)
   python scripts/run_pipeline.py
   ```

This downloads the arXiv dataset from Kaggle, filters it by date (2021+), chunks the abstracts, creates embeddings using BAAI/bge-small-en-v1.5, and uploads them to Pinecone.

## Usage

### Interactive Mode

Run the main application for interactive queries:

```bash
python main.py
```

The system will:
- Load all models and components
- Generate a workflow visualization
- Provide an interactive prompt for questions

### Batch Processing

Process multiple queries from the `prompts.txt` file:

```bash
python batch_processor.py
```

Results are saved to `output/batch_results_[timestamp].csv` with comparisons across different approaches.

## Docker Setup

The project includes Docker support for easy deployment and development.

### Prerequisites for Docker

- Docker and Docker Compose installed
- API keys set in `.env` file

### Production Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Run specific service:**
   ```bash
   # Production service
   docker-compose up agentic-rag
   
   # Development service (with live code mounting)
   docker-compose --profile dev up agentic-rag-dev
   ```

3. **Run interactively:**
   ```bash
   docker-compose run --rm agentic-rag
   ```

### Manual Docker Build

1. **Build the image:**
   ```bash
   docker build -t agentic-rag .
   ```

2. **Run the container:**
   ```bash
   docker run -it --rm \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/output:/app/output \
     -v $(pwd)/.env:/app/.env:ro \
     -e PINECONE_API_KEY=$PINECONE_API_KEY \
     -e GEMINI_API_KEY=$GEMINI_API_KEY \
     agentic-rag
   ```

### Development with Docker

For development with live code changes:

```bash
# Start development environment
docker-compose --profile dev up agentic-rag-dev

# Or run specific commands
docker-compose run --rm agentic-rag-dev python download_filter_embed_upsert.py
```

## Configuration

The system is configured through `config/config.yaml`:

```yaml
# Model Configuration  
gemini_model_name: "gemini-2.0-flash"
fast_embed_name: "BAAI/bge-small-en-v1.5"
top_k: 5

# Database Configuration
pc_index: "abstract-index"
distance_metric: "cosine"

# Directory Configuration
data_dir: "data"
output_dir: "output"
embedding_cache_dir: "embedding_cache"

# Data Processing
date: "2021-10-01"              # Filter papers from this date
chunk_size: 2000                # Text chunk size
chunk_overlap: 200              # Overlap between chunks
min_text_len: 50                # Minimum text length

# Prompts (customizable for different use cases)
classification_prompt: |
  # Custom classification logic for determining ArXiv retrieval
rewrite_prompt: |
  # Custom rewrite logic for query enhancement
final_prompt: |
  # Custom final response generation prompt
```

## API Documentation

### Core Components

- **`RAGState`**: TypedDict defining workflow state with query, documents, response, etc.
- **`build_rag_graph()`**: Main workflow builder creating LangGraph state machine
- **`load_all_components()`**: Component initialization for models, embeddings, and connections
- **`visualize_graph()`**: Workflow visualization generator

### Key Node Functions

- **Classification Node**: `create_classify_node()` - Determines if ArXiv retrieval is needed
- **Rewrite Node**: `create_rewrite_node()` - Generates query variations using Gemini
- **Retrieval Node**: `create_retrieve_node()` - Semantic document search with Pinecone
- **Simplification Node**: `create_simplify_abstracts_node()` - Gemini-powered text simplification
- **Generation Node**: `create_generate_response_node()` - Final answer synthesis
- **Direct Answer Node**: `create_direct_response_node()` - Direct Gemini responses for non-ArXiv queries

### Workflow Routing

- **`route_based_on_classification()`**: Conditional routing based on classification results
- **Conditional Edges**: Smart routing between ArXiv-enhanced and direct response paths

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) for research papers
- [Google AI](https://ai.google.dev/) for Gemini 2.0 Flash integration
- [Pinecone](https://www.pinecone.io/) for vector database services
- [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow orchestration
- [BAAI](https://huggingface.co/BAAI) for BGE embedding models
- [Hugging Face](https://huggingface.co/) for model hosting and transformers ecosystem

## Testing

The project includes a comprehensive test suite using pytest:

```bash
# Run all tests
pytest

# Run specific test modules
pytest test/test_core/
pytest test/test_connections/
pytest test/test_loaders/

# Run with coverage
pytest --cov=src

# Run unit tests with the actual RAG system
python test/unit_test.py
```

### Test Structure

- **`test/test_core/`**: Core RAG functionality tests
- **`test/test_connections/`**: Pinecone and Gemini connection tests  
- **`test/test_loaders/`**: Data loading and embedding tests
- **`test/test_models/`**: Model loading tests
- **`test/conftest.py`**: Shared test fixtures and configuration

## Additional Features

### RAG Comparison Framework

The `rag_comparison/` directory contains comparative implementations for research purposes:

- **Simple RAG**: Basic retrieval-augmented generation
- **Advanced RAG**: Multi-query retrieval with classification
- **Batch Processing**: Comparative analysis across different RAG approaches

### Legacy Scripts

The project includes standalone scripts for compatibility:

- **`download_filter_embed_upsert.py`**: Legacy data processing pipeline
- **`batch_processor.py`**: Multi-approach comparison tool
- **`unit_test_prep.py`**: Test data preparation utilities

These provide alternative entry points and comparison baselines for the modular system.