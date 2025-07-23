# Simple RAG System

A lightweight Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, and FAISS for document question-answering.

![RAG System Diagram](https://miro.medium.com/v2/resize:fit:1400/1*Q8D0k3QOQZQZQZQZQZQZQ.png)

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

```

Key improvements made:
1. Added visual diagram placeholder
2. Organized features into clear bullet points
3. Added separate troubleshooting sections
4. Included configuration examples
5. Improved formatting for better readability
6. Added license section (you may want to add a LICENSE file)
7. Made the structure more scannable with clear section headers

You may want to:
1. Add a real diagram image URL
2. Include a LICENSE file if using MIT license
3. Add contribution guidelines if open source
4. Include system requirements based on your testing
