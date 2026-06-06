import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.prompts import PromptTemplate

# ---------------- ENV ----------------
load_dotenv()
DB_DIR = "rag_db"

# ---------------- RESET DB ----------------
def reset_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

# ✅ RESET DB ON EVERY PAGE LOAD (ONCE PER SESSION)
if "db_reset_done" not in st.session_state:
    reset_db()
    st.session_state.db_reset_done = True

# ---------------- SESSION STATE ----------------
if "query" not in st.session_state:
    st.session_state.query = ""

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ---------------- EMBEDDINGS ----------------
embedding = FastEmbedEmbeddings()

# ---------------- SPLITTER ----------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ---------------- PROMPT ----------------
prompt = PromptTemplate.from_template(
"""
Answer ONLY from the context below.

Rules:
- Return exact answer only
- If name → only name
- If not found → "Not in document"

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------- INGEST ----------------
def ingest(folder):
    docs_all = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)

            loader = PDFPlumberLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata["page"] = d.metadata.get("page", 0)

            docs_all.extend(docs)

    if not docs_all:
        return "❌ No PDFs found"

    chunks = splitter.split_documents(docs_all)

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    vectordb.add_documents(chunks)
    vectordb.persist()

    return f"✅ DB ready with {len(chunks)} chunks"


# ---------------- ASK ----------------
def ask(query):
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    docs_with_scores = vectordb.similarity_search_with_score(query, k=5)

    best_doc, best_score = sorted(docs_with_scores, key=lambda x: x[1])[0]

    context = best_doc.page_content.strip()

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    best_page = best_doc.metadata.get("page", "N/A")

    return answer, best_page, best_doc


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Fast RAG App", layout="wide")

st.title("📄 Fast RAG PDF Q&A")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join("uploads", file.name), "wb") as f:
            f.write(file.getbuffer())

    if st.button(" Process PDFs"):
        with st.spinner("Processing..."):
            msg = ingest("uploads")
            st.session_state.processed = True
            st.success(msg)

# Ask question ONLY after processing
if st.session_state.processed:
    st.session_state.query = st.text_input(
        "❓ Ask a question",
        value=st.session_state.query
    )

    if st.session_state.query:
        with st.spinner("Thinking..."):
            answer, page, doc = ask(st.session_state.query)

        st.subheader("✅ Answer")
        st.write(answer)

        st.subheader("📄 Page")
        st.write(page)

        with st.expander("🔍 Best Match Chunk"):
            st.write(doc.page_content[:500])

        # 🔄 Ask Again Button
        if st.button("🔄 Ask Another Question"):
            st.session_state.query = ""
            st.rerun()