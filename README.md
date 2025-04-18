# RAG---Ollama

A Retrieval-Augmented Generation (RAG) application that uses local Ollama models to answer questions based on your documents. This project demonstrates how to build a complete RAG pipeline with purely local, self-hosted components.

## Architecture

![RAG Architecture](https://www.deepchecks.com/wp-content/uploads/2024/10/img-rag-architecture-model.jpg)
> Source: https://www.deepchecks.com/glossary/rag-architecture/

---
The application consists of three main components:
1. **RAG Application** - Python backend using llama-index for document processing and retrieval (runs as a local Python script)
2. **ChromaDB** - Vector database for storing embeddings (runs in Docker)
3. **Ollama** - Local LLM server providing embedding and text generation capabilities (runs as a local service)

## Prerequisites

- Docker (only for running ChromaDB)
- Python 3.13+
- Ollama installed locally

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG-Ollama.git
   cd RAG-Ollama
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install and start Ollama:
   - Follow instructions at [Ollama.com](https://ollama.com)
   - Pull required models:
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3.2
     ```

4. Start ChromaDB using Docker:
   ```bash
   docker-compose up -d
   ```

5. Update the `.env` file to point to your local ChromaDB:
   ```
   CHROMA_DB_HOST=localhost
   ```

6. Place your PDF documents in the `assets` folder

7. Run the application:
   ```bash
   python src/app.py
   ```

## Usage

Once running, the application will:
1. Process any PDFs in the `assets` folder
2. Create or update the vector database
3. Start an interactive command line interface

You can then ask questions about your documents, and the application will use RAG to generate relevant answers.

Example:
```
How can I help you?
What is the main benefit of retrieval augmented generation?

Searching for answer...
Based on the documents, the main benefit of Retrieval-Augmented Generation (RAG) is that it helps reduce hallucinations in large language models by grounding responses in retrieved documents. This makes the system more accurate and trustworthy.
```

## Configuration

### Ollama Models

The application uses these Ollama models by default:
- `nomic-embed-text` for embeddings
- `llama3.2` for text generation

To change models, modify the `get_embedding_model()` and `get_llm()` methods in `src/engine/ChatEngine.py`.

### Vector Database

ChromaDB settings can be adjusted in the `docker-compose.yml` file.

## Project Structure

```
RAG-Ollama/
├── assets/                  # Place your PDFs here
├── src/
│   ├── app.py               # Main application entry point
│   ├── engine/
│   │   └── ChatEngine.py    # Chat engine implementation
│   ├── utils/
│   │   └── file_reader.py   # Document processing utilities
│   └── vectorstore/
│       └── ingestion.py     # Vector database ingestion
├── chroma/                  # ChromaDB persistent storage (will be mounted by the docker-compose.yml)
├── docker-compose.yml       # Docker configuration for ChromaDB
└── requirements.txt         # Python dependencies
```

## Features

- 100% local and private RAG pipeline
- PDF document processing
- Interactive chat interface
- Persistent vector database storage

## License

[MIT License](LICENSE)
