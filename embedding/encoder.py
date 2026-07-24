from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_ID

def get_embeddings():
    """
    Initializes and returns the BAAI/bge-small-en-v1.5 embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
