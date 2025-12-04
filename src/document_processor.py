import os
import tempfile
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using Unstructured library"""
        try:
            elements = partition_pdf(
                filename=pdf_path,
                strategy="fast",
                extract_images=False,
            )
            
            text_elements = []
            for i, element in enumerate(elements):
                if hasattr(element, 'text') and element.text.strip():
                    text_elements.append({
                        'id': i,
                        'text': element.text.strip(),
                        'type': type(element).__name__
                    })
            
            return text_elements
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def chunk_document(self, text_elements: List[Dict[str, Any]]) -> List[Document]:
        """Split document into chunks for processing"""
        full_text = "\n\n".join([elem['text'] for elem in text_elements])
        
        # Create LangChain documents
        documents = [Document(page_content=full_text, metadata={"source": "pdf"})]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        return chunks
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Main method to process PDF file"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Extract text
            text_elements = self.extract_text_from_pdf(tmp_path)
            
            if not text_elements:
                raise Exception("No text content found in PDF")
            
            # Chunk document
            chunks = self.chunk_document(text_elements)
            
            return chunks
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)