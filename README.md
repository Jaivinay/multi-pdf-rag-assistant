````markdown
# Enterprise RAG Assistant

A local Retrieval-Augmented Generation application that allows users to upload multiple PDF files and ask questions from the uploaded knowledge base. The system retrieves relevant information from the documents and generates concise answers using a local language model.

## Overview

This project demonstrates an end-to-end RAG workflow using Python, Streamlit, LangChain, FAISS, Ollama, and Llama 3.2.

The application allows users to upload PDF documents, extract text, split the content into smaller chunks, generate embeddings, store them in a FAISS vector database, retrieve relevant chunks based on user questions, and generate answers using a local LLM.

## Tech Stack

Python, Streamlit, LangChain, FAISS, Ollama, Llama 3.2, Nomic Embed Text, PyPDFLoader

## How It Works

The user uploads PDF files through the Streamlit interface. The application extracts text from the PDFs using PyPDFLoader and splits the text into smaller overlapping chunks. Each chunk is converted into an embedding using the Nomic Embed Text model. These embeddings are stored in FAISS for similarity search.

When a user asks a question, the question is also converted into an embedding. FAISS compares it with the stored document embeddings and retrieves the most relevant chunks. These chunks are passed as context to Llama 3.2 through Ollama, which generates a concise answer based on the retrieved information.

The application also saves the FAISS index and document metadata locally so the knowledge base can be reused without reprocessing the PDFs. Query logs are stored with question, answer, retrieved sources, response time, and model details.

## Architecture

PDF Upload  
→ Text Extraction  
→ Document Chunking  
→ Embedding Generation  
→ FAISS Vector Store  
→ Similarity Search  
→ Retrieved Context  
→ Llama 3.2  
→ Final Answer

## Features

- Upload and process multiple PDF files
- Extract text from PDF documents
- Split documents into smaller overlapping chunks
- Generate embeddings using Nomic Embed Text
- Store and retrieve vectors using FAISS
- Generate answers using local Llama 3.2 through Ollama
- View retrieved sources and page references
- Save FAISS index for reuse
- Track query logs and response latency
- Reset knowledge base when needed

## Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Jaivinay/multi-pdf-rag-assistant.git
cd multi-pdf-rag-assistant
````

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama and pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Run the application:

```bash
streamlit run app.py
```

Open the app in your browser:

```text
http://localhost:8501
```

## Example Questions

What is Amazon S3?
Explain AWS Glue.
What is overfitting?
Explain SQL window functions.
What is Python?

## Key Concepts Demonstrated

Retrieval-Augmented Generation, Large Language Models, vector embeddings, semantic search, FAISS vector database, prompt engineering, local LLM usage with Ollama, Streamlit application development, persistent vector storage, and query logging.

