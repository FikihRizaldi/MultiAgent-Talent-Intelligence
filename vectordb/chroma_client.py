import chromadb
from chromadb.config import Settings
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "chroma_db")

class ChromaDBClient:
    def __init__(self):
        # Using persistent client for ChromaDB
        self.client = chromadb.PersistentClient(path=DB_PATH)
        
    def get_or_create_collection(self, collection_name: str):
        return self.client.get_or_create_collection(name=collection_name)

chroma_client = ChromaDBClient()
