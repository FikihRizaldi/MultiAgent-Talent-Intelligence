from typing import List, Dict
from vectordb.chroma_client import chroma_client
from embedding.encoder import encoder

class RAGRetriever:
    def __init__(self, collection_name: str = "company_knowledge"):
        self.collection = chroma_client.get_or_create_collection(collection_name)
        
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Embeds and stores documents in ChromaDB.
        """
        embeddings = encoder.encode(documents)
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(documents)} documents to ChromaDB.")
        
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Performs semantic search and returns top-k matching documents.
        """
        query_embedding = encoder.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for idx in range(len(results['documents'][0])):
                formatted_results.append({
                    "id": results['ids'][0][idx],
                    "document": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx],
                    "distance": results['distances'][0][idx]
                })
                
        return formatted_results

retriever = RAGRetriever()
