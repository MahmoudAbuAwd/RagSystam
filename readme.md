# Simple RAG System

A lightweight Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, and FAISS for document question-answering.

![RAG Architecture Diagram](https://www.dailydoseofds.com/content/images/2024/11/ragdiagram-ezgif.com-resize.gif)

*A visual overview of the RAG system architecture*
## Features

- 📄 Process PDF, TXT, and CSV documents
- 💬 Interactive chat interface (Streamlit)
- ⚡ Local LLM support via Ollama
- 🔍 Semantic search with FAISS vector store
- 🛠️ Configurable chunking and retrieval parameters

## Prerequisites

- Python 3.8+
- Ollama installed ([Installation Guide](https://ollama.ai))
- At least 8GB RAM (recommended)

## Installation

1. **Set up Ollama**:
   ```bash
   ollama pull llama3.2:1b
   ollama serve
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```
1. Upload documents via the sidebar
2. Click "Process Documents"
3. Chat with your documents in the main interface

### Command Line Interface
```bash
python rag_system.py
```
- Place documents in the `data/` folder
- The system will automatically process them
- Enter questions at the prompt

## Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = "llama3.2:1b"  # Ollama model name
TEMPERATURE = 0.7           # Creativity control

# Document processing
CHUNK_SIZE = 1000           # Character count per chunk
CHUNK_OVERLAP = 200         # Overlap between chunks

# Retrieval
K = 3                       # Number of chunks to retrieve
```

## Supported File Formats

- PDF (.pdf)
- Plain Text (.txt)
- CSV (.csv)

## Example Queries

- "Summarize this document in 3 bullet points"
- "What are the key findings?"
- "Explain the methodology section"
- "List all recommendations mentioned"

## Troubleshooting

**Ollama Issues**
- Ensure `ollama serve` is running
- Verify model installation: `ollama list`
- Try a smaller model if facing memory issues

**Processing Errors**
- Check file permissions
- Ensure documents contain extractable text
- Reduce chunk size in config for large documents

## Project Structure

```
rag-system/
├── README.md
├── requirements.txt
├── .gitignore
├── rag_system.py          # CLI interface
├── app.py                 # Streamlit web app
├── config.py              # Configuration
├── data/                  # Document storage (CLI)
└── cache/                 # FAISS vector store cache
```

## License

MIT License - See [LICENSE](LICENSE) for details

