# Enterprise Multi-PDF RAG Assistant

## Overview

Enterprise Multi-PDF RAG Assistant is a Retrieval-Augmented Generation (RAG) application built using Python, Streamlit, LangChain, FAISS, Ollama, and Llama 3.2. The application enables users to upload multiple PDF documents, build a searchable knowledge base, and interact with the uploaded content through natural language.

The system combines semantic search with a local Large Language Model (LLM) to retrieve relevant information from uploaded documents and generate context-aware responses. The project demonstrates the complete RAG workflow, including document ingestion, text preprocessing, embedding generation, vector storage, semantic retrieval, prompt engineering, and local LLM inference.

---

## Key Features

- Multi-PDF document upload and processing
- Automatic text extraction using PyPDFLoader
- Recursive document chunking with configurable overlap
- Semantic embeddings using Nomic Embed Text
- FAISS vector database for similarity search
- Local Llama 3.2 inference using Ollama
- Retrieval-Augmented Generation (RAG)
- Streamlit-based conversational interface
- Source document and page reference display
- Persistent FAISS index and document metadata
- Chat history persistence
- Query logging and response latency tracking
- Knowledge base reset functionality

---

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| User Interface | Streamlit |
| AI Framework | LangChain |
| Vector Database | FAISS |
| Embedding Model | Nomic Embed Text |
| Large Language Model | Llama 3.2 |
| Local Model Runtime | Ollama |
| Document Processing | PyPDFLoader |

---

## System Architecture

```
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Document Chunking
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Vector Database
      │
      ▼
Semantic Similarity Search
      │
      ▼
Relevant Context Retrieval
      │
      ▼
Llama 3.2 (Ollama)
      │
      ▼
AI Generated Response
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/Jaivinay/multi-pdf-rag-assistant.git
cd multi-pdf-rag-assistant
```

Create and activate a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

Install Ollama and download the required models.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Run the application.

```bash
streamlit run app.py
```

Open the application at:

**http://localhost:8501**

---

## Project Workflow

1. Upload one or more PDF documents.
2. Extract text from the uploaded documents.
3. Split the extracted text into overlapping chunks.
4. Generate vector embeddings for each chunk.
5. Store embeddings in a FAISS vector database.
6. Convert the user question into an embedding.
7. Retrieve the most relevant document chunks using semantic similarity.
8. Pass the retrieved context to Llama 3.2 through Ollama.
9. Generate an accurate, context-aware response with supporting document references.

---

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Embeddings
- Semantic Search
- Vector Databases
- Prompt Engineering
- Local AI Model Deployment
- LangChain
- FAISS
- Streamlit Application Development

---

## Future Enhancements

- Support additional document formats (DOCX, TXT, CSV)
- Streaming response generation
- Conversation memory
- Hybrid search and reranking
- Authentication and user management
- Cloud deployment
- Analytics dashboard
- Resume and Job Description comparison
- AI Interview Assistant

---

## Author

**Jaivinay Gudiveka**

GitHub: https://github.com/Jaivinay

LinkedIn: https://www.linkedin.com/in/jaivinay-gudiveka