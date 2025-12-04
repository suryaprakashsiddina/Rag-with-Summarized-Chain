import streamlit as st
import os
import time
from src.document_processor import DocumentProcessor
from src.summarizer import Summarizer
from src.vector_store import MultiVectorStore
from src.rag_chain import SummarizedRAGChain
from src.utils import *

# Page configuration
st.set_page_config(
    page_title="Academic RAG with Summarized Chain",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    # .info-box {
    #     background-color: #f0f2f6;
    #     padding: 1rem;
    #     border-radius: 0.5rem;
    #     margin: 1rem 0;
    # }
    # .info-box {
    #     background: linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%);
    #     padding: 1.5rem;
    #     border-radius: 0.75rem;
    #     margin: 1rem 0;
    #     color: #ffffff;
    #     border-left: 4px solid #1f77b4;
    #     box-shadow: 0 4px 15px rgba(31, 119, 180, 0.2);
    #     position: relative;
    #     overflow: hidden;
    # }

    # .info-box::before {
    #     content: '';
    #     position: absolute;
    #     top: 0;
    #     left: 0;
    #     right: 0;
    #     height: 2px;
    #     background: linear-gradient(90deg, #1f77b4, #ff6b6b);
    # }
    .info-box {
    background: #000000;
    padding: 1.25rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    color: #1f77b4;
    border: 1px solid #1f77b4;
    position: relative;
    transition: all 0.3s ease;
}

.info-box:hover {
    background: #0a0a0a;
    box-shadow: 0 0 20px rgba(31, 119, 180, 0.3);
}

.info-box::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #1f77b4, transparent);
}
    
    /* ===== BEAUTIFUL CHAT MESSAGE STYLES ===== */
    /* User message styling */
    .user-message {
        # background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background: linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border: none;
        max-width: 85%;
        margin-left: auto;
        position: relative;
    }
    
    .user-message::before {
        content: "👤";
        position: absolute;
        left: -45px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.2rem;
        background: #667eea;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    
    /* Assistant message styling */
    .assistant-message {
        # background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        background: linear-gradient(135deg, #000000 0%, #1a1a2e 50%, #16213e 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
        border: none;
        max-width: 85%;
        margin-right: auto;
        position: relative;
    }
    
    .assistant-message::before {
        content: "🤖";
        position: absolute;
        right: -45px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.2rem;
        background: #f093fb;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Message header styling */
    .message-sender {
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Chat container improvements */
    .stChatMessage {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Smooth animations */
    .stChatMessage {
        animation: fadeInUp 0.5s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sources expander styling */
    .sources-expander {
        margin-top: 1rem;
    }
    
    .sources-expander .streamlit-expanderHeader {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.8rem 1rem;
    }
    
    .sources-expander .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #e1e5e9;
        border-radius: 0 0 10px 10px;
        margin-top: -1px;
    }
    
    /* Source item styling */
    .source-item {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .source-header {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }
    
    .source-content {
        color: #4a5568;
        font-size: 0.9rem;
        line-height: 1.4;
        background: #f7fafc;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #4299e1;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    load_css()
    
    # Header
    st.markdown('<div class="main-header">📚 Academic RAG with Summarized Chain</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Innovative RAG system</strong> that summarizes document chunks before embedding, 
    enabling more focused and context-rich responses for academic document analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key to use the LLM"
        )
        
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        st.header("Document Processing")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type="pdf",
            help="Upload an academic PDF document (thesis, report, paper)"
        )
        
        if uploaded_file:
            valid, message = validate_pdf_file(uploaded_file)
            if not valid:
                st.error(message)
            else:
                st.session_state.current_pdf = uploaded_file.name
                
                if st.button("Process Document", type="primary"):
                    process_document(uploaded_file)
        
        # Display system info
        if st.session_state.processed_docs:
            st.success("✅ Document processed and ready for queries")
            
            if st.session_state.vector_store:
                store_info = st.session_state.vector_store.get_store_info()
                st.write("**Vector Stores:**")
                st.json(store_info)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">💬 Chat with Your Document</div>', unsafe_allow_html=True)
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        if st.session_state.processed_docs:
            if question := st.chat_input("Ask a question about your document..."):
                # Add user message to chat
                user_message = format_chat_message("user", question)
                st.session_state.chat_history.append(user_message)
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="message-sender">You</div>
                        <div class="message-content">{question}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching document..."):
                        response = query_document(question)
                    
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-sender">Assistant</div>
                        <div class="message-content">{response["answer"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources if available
                    if response.get("sources"):
                        with st.expander("📚 View Sources"):
                            st.markdown('<div class="sources-expander">', unsafe_allow_html=True)
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"""
                                <div class="source-item">
                                    <div class="source-header">Source {i+1}</div>
                                    <div class="source-content">
                                        {source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                
                # Add assistant message to chat history
                assistant_message = format_chat_message(
                    "assistant", 
                    response["answer"], 
                    response.get("sources", [])
                )
                st.session_state.chat_history.append(assistant_message)
        
        else:
            st.info("👈 Please upload and process a PDF document to start chatting")
    
    with col2:
        st.markdown('<div class="sub-header">🔍 System Overview</div>', unsafe_allow_html=True)
        
        st.write("""
        **Summarized Chain Process:**
        1. **Document Extraction** - Extract text from PDF using Unstructured
        2. **Chunk Processing** - Split document into manageable chunks
        3. **Chunk Summarization** - Generate concise summaries using Groq LLM
        4. **Vector Embedding** - Create embeddings from summarized chunks
        5. **Intelligent Retrieval** - Retrieve relevant original chunks based on summary matches
        6. **Contextual Response** - Generate accurate answers using retrieved context
        """)
        
        if st.session_state.processed_docs:
            st.success("System Ready")
            st.write(f"**Current Document:** {st.session_state.current_pdf}")
            
            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

def process_document(uploaded_file):
    """Process the uploaded PDF document"""
    try:
        with st.spinner("Processing document..."):
            # Step 1: Document processing
            doc_processor = DocumentProcessor()
            chunks = doc_processor.process_pdf(uploaded_file)
            
            if not chunks:
                st.error("No content extracted from PDF")
                return
            
            # Step 2: Summarization
            summarizer = Summarizer()
            summarized_chunks = summarizer.create_summarized_chunks(chunks)
            
            # Step 3: Vector store initialization
            vector_store = MultiVectorStore()
            
            # Create memory store with summarized chunks
            memory_store = vector_store.create_memory_store(summarized_chunks, "summarized_chunks")
            
            # Also store original chunks in Chroma for reference
            vector_store.add_to_chroma(chunks, "original_chunks")
            
            # Step 4: RAG chain setup
            rag_system = SummarizedRAGChain()
            qa_chain = rag_system.create_retrieval_chain(memory_store)
            
            # Store in session state
            st.session_state.vector_store = vector_store
            st.session_state.rag_chain = qa_chain
            st.session_state.processed_docs = True
            
            st.success(f"✅ Successfully processed {len(chunks)} chunks with {len(summarized_chunks)} summaries")
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")

def query_document(question: str):
    """Query the processed document"""
    try:
        if not st.session_state.rag_chain:
            return {"answer": "System not initialized. Please process a document first.", "sources": []}
        
        rag_system = SummarizedRAGChain()
        response = rag_system.query_documents(question, st.session_state.rag_chain)
        
        return response
        
    except Exception as e:
        return {"answer": f"Error querying document: {str(e)}", "sources": []}

if __name__ == "__main__":
    main()