from typing import List, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from langchain_community.llms import Groq
from langchain_groq import ChatGroq
from config import Config

class SummarizedRAGChain:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        # Define RAG prompt template
        self.qa_prompt = PromptTemplate(
            template="""
            You are an AI assistant designed to help with academic document analysis.
            Use the following context from a PDF document to answer the user's question.
            
            CONTEXT:
            {context}
            
            QUESTION: {question}
            
            INSTRUCTIONS:
            1. Answer based ONLY on the provided context
            2. If the context doesn't contain relevant information, say so
            3. Be precise and concise in your answers
            4. Focus on academic and technical accuracy
            5. If referring to specific sections, mention them clearly
            
            ANSWER:""",
            input_variables=["context", "question"]
        )
    
    def create_retrieval_chain(self, vector_store, store_type: str = "memory"):
        """Create retrieval QA chain"""
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.TOP_K_RETRIEVAL}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.qa_prompt},
            return_source_documents=True
        )
        
        return qa_chain
    
    def query_documents(self, question: str, qa_chain) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            result = qa_chain({"query": question})
            
            response = {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "success": True
            }
            
            return response
            
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "success": False
            }