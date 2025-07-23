# Configuration file for RAG System

# LLM Settings
LLM_MODEL = "llama3.2:1b"

# Embedding Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text Splitting Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
RETRIEVAL_K = 3  # Number of chunks to retrieve

# Paths
DATA_FOLDER = "data"
CACHE_FOLDER = "cache"

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.csv']