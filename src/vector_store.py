import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from config import Config

class MultiVectorStore:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.chroma_store = None
        self.memory_store = None
        self.current_stores = {}
    
    def initialize_chroma(self, persist_directory: str):
        """Initialize ChromaDB vector store"""
        self.chroma_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
    
    def create_memory_store(self, documents: List[Document], store_id: str):
        """Create in-memory vector store"""
        self.memory_store = FAISS.from_documents(
            documents, 
            self.embedding_model
        )
        self.current_stores[store_id] = self.memory_store
        return self.memory_store
    
    def add_to_chroma(self, documents: List[Document], collection_name: str):
        """Add documents to ChromaDB"""
        if self.chroma_store is None:
            self.chroma_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=Config.CHROMA_PERSIST_DIR
            )
        else:
            self.chroma_store.add_documents(documents)
        
        self.chroma_store.persist()
    
    def similarity_search(self, query: str, k: int = 3, store_type: str = "memory"):
        """Search for similar documents"""
        if store_type == "chroma" and self.chroma_store:
            return self.chroma_store.similarity_search(query, k=k)
        elif store_type == "memory" and self.memory_store:
            return self.memory_store.similarity_search(query, k=k)
        else:
            raise Exception("Vector store not initialized")
    
    def similarity_search_with_score(self, query: str, k: int = 3, store_type: str = "memory"):
        """Search for similar documents with scores"""
        if store_type == "chroma" and self.chroma_store:
            return self.chroma_store.similarity_search_with_score(query, k=k)
        elif store_type == "memory" and self.memory_store:
            return self.memory_store.similarity_search_with_score(query, k=k)
        else:
            raise Exception("Vector store not initialized")
    
    def get_store_info(self):
        """Get information about current vector stores"""
        info = {
            "chroma_initialized": self.chroma_store is not None,
            "memory_initialized": self.memory_store is not None,
            "current_stores": list(self.current_stores.keys())
        }
        return info