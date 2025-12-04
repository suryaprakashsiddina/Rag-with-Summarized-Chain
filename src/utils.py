import streamlit as st
from typing import List, Dict, Any
from langchain.schema import Document

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None

def format_chat_message(role: str, content: str, sources: List[Document] = None):
    """Format chat message with sources"""
    message = {
        "role": role,
        "content": content,
        "sources": sources if sources else []
    }
    return message

def display_chat_history():
    """Display chat history in Streamlit with beautiful styling"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-sender">You</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-sender">Assistant</div>
                    <div class="message-content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        st.markdown('<div class="sources-expander">', unsafe_allow_html=True)
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-item">
                                <div class="source-header">Source {i+1}</div>
                                <div class="source-content">
                                    {source.page_content[:500] + "..." if len(source.page_content) > 500 else source.page_content}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

def validate_pdf_file(uploaded_file):
    """Validate uploaded PDF file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.type != "application/pdf":
        return False, "Please upload a PDF file"
    
    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
        return False, "File size too large. Please upload a PDF smaller than 50MB"
    
    return True, "Valid PDF file"