from typing import List, Dict
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
# from langchain_community.llms import Groq
from langchain_groq import ChatGroq # Or just Groq, depending on your specific use case
from config import Config

class Summarizer:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL
        )
        
        # Define summarization prompts
        self.map_prompt = PromptTemplate(
            template="""
            Write a concise summary of the following text excerpt from an academic document.
            Focus on the key points, findings, methodologies, and conclusions.
            
            TEXT: {text}
            
            CONCISE SUMMARY:""",
            input_variables=["text"]
        )
        
        self.combine_prompt = PromptTemplate(
            template="""
            Create a comprehensive summary by synthesizing the following excerpts and their summaries.
            The summary should capture the main themes, methodologies, results, and conclusions.
            
            SUMMARIES:
            {text}
            
            COMPREHENSIVE SUMMARY:""",
            input_variables=["text"]
        )
    
    def summarize_chunks(self, chunks: List[Document]) -> List[Document]:
        """Summarize document chunks using LLM"""
        try:
            # Create summarization chain
            summary_chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=self.map_prompt,
                combine_prompt=self.combine_prompt,
                verbose=False
            )
            
            # Generate summaries
            summary_result = summary_chain.run(chunks)
            
            # Create summary documents
            summary_doc = Document(
                page_content=summary_result,
                metadata={"type": "summary", "source": "llm_summarization"}
            )
            
            return [summary_doc]
            
        except Exception as e:
            raise Exception(f"Error in summarization: {str(e)}")
    
    def create_summarized_chunks(self, chunks: List[Document]) -> List[Document]:
        """Create summarized versions of chunks for embedding"""
        summarized_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Create a prompt for chunk summarization
                chunk_prompt = f"""
                Summarize the following text chunk from an academic document in 2-3 sentences.
                Focus on the core content and key information:
                
                {chunk.page_content}
                
                Summary:"""
                
                # Generate summary
                summary = self.llm.invoke(chunk_prompt)
                
                # Create new document with summary
                summary_doc = Document(
                    page_content=summary.strip(),
                    metadata=chunk.metadata.copy()
                )
                summary_doc.metadata["is_summary"] = True
                summary_doc.metadata["original_chunk_id"] = i
                
                summarized_chunks.append(summary_doc)
                
            except Exception as e:
                print(f"Error summarizing chunk {i}: {str(e)}")
                # Fallback: use original chunk
                summarized_chunks.append(chunk)
        
        return summarized_chunks