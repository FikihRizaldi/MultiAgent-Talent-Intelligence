from langchain_community.vectorstores import Chroma
from embedding.encoder import get_embeddings
from config import CHROMA_DB_DIR
import os

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(CHROMA_DB_DIR):
            os.makedirs(CHROMA_DB_DIR)
        embeddings = get_embeddings()
        _vectorstore = Chroma(
            collection_name="enterprise_knowledge",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
    return _vectorstore

def get_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})
