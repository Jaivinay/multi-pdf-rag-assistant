import os
import json
import shutil
import time
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from ollama import chat


PDF_INFO_FILE = "pdf_info.json"
FAISS_INDEX_DIR = "faiss_index"
DOCUMENTS_DIR = "documents"
LOG_DIR = "logs"
QUERY_LOG_FILE = "logs/query_logs.jsonl"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="📚",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def save_pdf_info(file_names, pages, chunks):
    info = {
        "files": file_names,
        "pages": pages,
        "chunks": chunks,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(PDF_INFO_FILE, "w") as f:
        json.dump(info, f, indent=4)


def load_pdf_info():
    if os.path.exists(PDF_INFO_FILE):
        try:
            with open(PDF_INFO_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def load_faiss_index():
    if os.path.exists(FAISS_INDEX_DIR):
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

            vector_store = FAISS.load_local(
                FAISS_INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )

            return vector_store

        except Exception as e:
            st.warning(f"Could not load saved FAISS index: {e}")
            return None

    return None


def log_query(question, answer, sources, latency_seconds):
    os.makedirs(LOG_DIR, exist_ok=True)

    log_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": [
            {
                "file": source["file"],
                "page": source["page"]
            }
            for source in sources
        ],
        "latency_seconds": latency_seconds,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }

    with open(QUERY_LOG_FILE, "a") as f:
        f.write(json.dumps(log_record) + "\n")


def process_pdfs(uploaded_files):
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    all_documents = []

    for uploaded_file in uploaded_files:
        pdf_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(all_documents)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_store.save_local(FAISS_INDEX_DIR)

    return vector_store, len(all_documents), len(chunks)


def is_greeting(question):
    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hai",
        "good morning",
        "good afternoon",
        "good evening"

    ]

    return question.lower().strip() in greetings


def generate_greeting_answer():
    return """
Hello! 👋

I'm your enterprise-style RAG assistant.

You can ask me questions from the processed knowledge base, and I will answer with practical, source-grounded responses.
"""


def generate_rag_answer(question, vector_store):
    docs = vector_store.similarity_search(
        question,
        k=4
    )

    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")

        context_parts.append(
            f"Source: {source}, Page: {page}\nContent:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an enterprise AI assistant used by internal business and technology teams.

Your job is to answer the user's question using only the provided knowledge context.

Rules:
1. Answer directly and naturally.
2. Do not mention PDFs, documents, uploaded files, or context.
3. Do not say "according to", "based on", "from the PDF", or "the document says".
4. Keep the answer concise, practical, and interview-friendly.
5. If the answer is not available, say:
   "I don't have enough information to answer that confidently."
6. Use bullets when useful.
7. Do not invent facts.
8. Prefer clear business/technical explanation.

Knowledge Context:
{context}

User Question:
{question}

Final Answer:
"""

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    sources = []

    for doc in docs:
        sources.append(
            {
                "file": doc.metadata.get("source", "Unknown file"),
                "page": doc.metadata.get("page", "Unknown page"),
                "content": doc.page_content
            }
        )

    return answer, sources


if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_faiss_index()

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = st.session_state.vector_store is not None

if "file_names" not in st.session_state:
    st.session_state.file_names = []

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


pdf_info = load_pdf_info()

if pdf_info and st.session_state.pdf_processed:
    st.session_state.file_names = pdf_info.get("files", [])
    st.session_state.total_pages = pdf_info.get("pages", 0)
    st.session_state.total_chunks = pdf_info.get("chunks", 0)


with st.sidebar:
    st.header("📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload up to 5 PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Please upload only up to 5 PDFs.")
        else:
            if st.button("Process Knowledge Base"):
                with st.spinner("Processing PDFs, creating chunks, embeddings, and FAISS index..."):
                    vector_store, total_pages, total_chunks = process_pdfs(uploaded_files)

                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    st.session_state.file_names = [file.name for file in uploaded_files]
                    st.session_state.total_pages = total_pages
                    st.session_state.total_chunks = total_chunks
                    st.session_state.messages = []

                    save_pdf_info(
                        st.session_state.file_names,
                        total_pages,
                        total_chunks
                    )

                st.success("Knowledge base processed successfully!")

    st.divider()

    st.subheader("📊 System Status")

    if st.session_state.pdf_processed:
        st.success("Ready")

        st.write("Files:")
        for file_name in st.session_state.file_names:
            st.write(f"• {file_name}")

        st.write(f"Total Pages: {st.session_state.total_pages}")
        st.write(f"Total Chunks: {st.session_state.total_chunks}")
        st.write(f"LLM: {LLM_MODEL}")
        st.write(f"Embeddings: {EMBEDDING_MODEL}")

    else:
        st.warning("No knowledge base processed yet.")

    st.divider()

    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Knowledge Base"):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.pdf_processed = False
        st.session_state.file_names = []
        st.session_state.total_pages = 0
        st.session_state.total_chunks = 0

        if os.path.exists(PDF_INFO_FILE):
            os.remove(PDF_INFO_FILE)

        if os.path.exists(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)

        st.rerun()


st.markdown(
    '<div class="main-title">📚 Enterprise RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Source-grounded AI assistant with retrieval, latency tracking, and enterprise-style query logging.</div>',
    unsafe_allow_html=True
)

if not st.session_state.pdf_processed:
    st.info("Upload PDFs from the sidebar and click **Process Knowledge Base** to start.")
else:
    st.success("Knowledge base is ready. Ask questions below.")


with st.expander("💡 Example Questions"):
    st.write("- What is Amazon S3?")
    st.write("- Explain AWS Glue in simple terms.")
    st.write("- What is Lambda used for?")
    st.write("- What is overfitting?")
    st.write("- Explain SQL window functions.")
    st.write("- How should I explain this in an interview?")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Retrieved Sources"):
                for i, source in enumerate(message["sources"]):
                    st.write(f"Source {i + 1}")
                    st.write(f"File: {source['file']}")
                    st.write(f"Page: {source['page']}")
                    st.write(source["content"])
                    st.write("---------------------")


user_question = st.chat_input("Ask a question...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if is_greeting(user_question):
            answer = generate_greeting_answer()
            sources = []
            latency_seconds = 0
            st.write(answer)

        else:
            if not st.session_state.pdf_processed or st.session_state.vector_store is None:
                answer = "Please upload and process the knowledge base first."
                sources = []
                latency_seconds = 0
                st.warning(answer)

            else:
                with st.spinner("Retrieving relevant context and generating answer..."):
                    start_time = time.time()

                    answer, sources = generate_rag_answer(
                        user_question,
                        st.session_state.vector_store
                    )

                    latency_seconds = round(time.time() - start_time, 2)

                    st.write(answer)

                    st.caption(f"Response time: {latency_seconds} seconds")

                    if sources:
                        with st.expander("View Retrieved Sources"):
                            for i, source in enumerate(sources):
                                st.write(f"Source {i + 1}")
                                st.write(f"File: {source['file']}")
                                st.write(f"Page: {source['page']}")
                                st.write(source["content"])
                                st.write("---------------------")

                    log_query(
                        user_question,
                        answer,
                        sources,
                        latency_seconds
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency_seconds": latency_seconds
        }
    )