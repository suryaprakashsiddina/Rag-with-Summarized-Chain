import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"  #"llama3-70b-8192"  # or "mixtral-8x7b-32768"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIR = "./vector_stores/chroma"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Summarization Configuration
    SUMMARY_LENGTH = 300
    SUMMARY_CHUNK_SIZE = 3000
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL = 3
    SIMILARITY_THRESHOLD = 0.7

config = Config()