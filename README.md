# **RAG-with-Summarized-Chain**

An advanced Retrieval-Augmented Generation (RAG) system designed to improve how users interact with large PDF documents through a responsive chat interface. This project introduces a *Summarized Chain* approach to significantly enhance retrieval quality and overall efficiency, especially in academic evaluation workflows.

---

## 🚀 **Overview**

Traditional RAG models retrieve information directly from document chunks, which often leads to noisy or incomplete context. This project solves that by introducing a **Summarized Chain**, where large text segments are first *summarized* using an LLM before generating embeddings.
These refined summaries produce a cleaner index, resulting in much more accurate and context-rich responses.

The system is ideal for:

* College professors evaluating lengthy project reports
* Students or researchers extracting insights from academic PDFs
* Anyone needing efficient, high-quality document-based Q&A

---

## 🧠 **Key Features**

* **Summarized Chain Approach**
  Large PDF text chunks are summarized using an LLM before embedding, improving the quality of retrieval and reducing noise.

* **RAG Chat Interface for PDFs**
  Users can upload any PDF and interact with it through a conversational interface.

* **Multi-Vector Store Support**
  Uses **ChromaDB** and **InMemory Vectorstores** to store both summarized embeddings and original text chunks.

* **High-Performance LLM Integration**
  Powered by **Groq LLM** through API calls for fast and accurate generation.

* **Document Parsing with Unstructured**
  Robust extraction of document content using the Unstructured library.

* **Streamlit Deployment**
  Fully interactive UI with real-time answering, making it user-friendly and easy to deploy.

---

## 🏗️ **Architecture**

1. **PDF Uploaded**
2. **Text extracted** using Unstructured
3. **Chunking** of large text segments
4. **Summarized Chain**

   * Each chunk summarized using Groq LLM
   * Summaries embedded to create a refined index
5. **Multi-vector retrieval**

   * Summary vectors retrieve relevant contexts
   * Original chunks fetched for accuracy
6. **Final answer generated** using RAG pipeline
7. **Displayed in Streamlit chat interface**

---

## 🛠️ **Tech Stack**

* **Frameworks & Libraries:**
  LangChain, Streamlit, Unstructured

* **LLM:**
  Groq API (LLM-based summarization + generation)

* **Vector Stores:**
  ChromaDB, InMemory Vectorstore

* **Languages:**
  Python

---

## 📌 **How It Works**

* Upload a PDF
* System preprocesses and summarizes large chunks
* Summaries are embedded for high-quality retrieval
* Retrieved summaries point to original high-context text
* A Groq-powered LLM generates the final output
* You get concise, accurate, context-rich answers

---

## 📂 **Use Cases**

* Academic project evaluation
* Research document analysis
* Report summarization & Q&A
* Knowledge extraction from large PDFs

---

## 📈 **Results**

* Improved retrieval accuracy due to summary-based embeddings
* More focused and context-rich answers than traditional RAG
* Reduced noise in retrieval
* Faster evaluation and feedback for academic reports

---

## ▶️ **How to Run**

1. Clone the repository

   ```bash
   git clone https://github.com/your-repo/Rag-with-Summarized-Chain.git
   cd Rag-with-Summarized-Chain
   ```
2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
3. Add your **Groq API key** to environment variables

   ```bash
   export GROQ_API_KEY="your_key"
   ```
4. Run the Streamlit app

   ```bash
   streamlit run app.py
   ```
