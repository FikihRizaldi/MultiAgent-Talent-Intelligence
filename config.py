import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "vectordb", "chroma_storage")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Models
LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

# RAG configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LangChain/LLM Settings
TEMPERATURE = 0.7
MAX_TOKENS = 512
