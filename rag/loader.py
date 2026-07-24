from embedding.encoder import get_embeddings
from vectordb.chroma_store import get_vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
import os
import glob

def load_company_knowledge():
    """
    Loads company knowledge JSON files, chunks them, and stores in Vector DB.
    """
    knowledge_path = os.path.join(DATA_DIR, "company_knowledge.json")
    if not os.path.exists(knowledge_path):
        print(f"Company knowledge dataset not found at {knowledge_path}")
        return
        
    loader = JSONLoader(
        file_path=knowledge_path,
        jq_schema='.[].content',
        text_content=False
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(splits)
    print(f"Successfully loaded {len(splits)} chunks into ChromaDB.")

if __name__ == "__main__":
    load_company_knowledge()
