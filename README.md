
# Multi-PDF RAG Assistant

A simple local RAG application that lets users upload multiple PDF files and ask questions from them using a local LLM.

## What It Does

This app reads uploaded PDFs, breaks them into smaller chunks, creates embeddings, stores them in FAISS, retrieves relevant information, and generates answers using Llama 3.2.

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Ollama
- Llama 3.2
- Nomic Embed Text

## Features

- Upload multiple PDF files
- Ask questions from uploaded documents
- Semantic search using FAISS
- Local LLM answers using Llama 3.2
- Chat-style interface
- Saved chat history
- Saved FAISS index
- View retrieved sources

## Project Flow

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Similarity Search
↓
Llama 3.2
↓
Final Answer

## How to Run

```bash
git clone https://github.com/Jaivinay/multi-pdf-rag-assistant.git
cd multi-pdf-rag-assistant
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Install Ollama and pull models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Run the app:

```bash
streamlit run app.py
```

## Key Concepts

* RAG
* Embeddings
* Vector Database
* Semantic Search
* FAISS
* LLM
* Prompt Engineering
* Document Question Answering

## Author

Jaivinay Gudiveka


LinkedIn: [https://www.linkedin.com/in/jaivinay-gudiveka](https://www.linkedin.com/in/jaivinay-gudiveka)
