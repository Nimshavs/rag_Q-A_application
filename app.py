import os
import streamlit as st
import logging

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.prompts import PromptTemplate

# ---------- ENV ----------
load_dotenv()

UPLOAD_FOLDER = "uploads"
DB_FOLDER = "db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ---------- LLM ----------
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

llm = load_llm()

# ---------- EMBEDDINGS ----------
@st.cache_resource
def load_embeddings():
    return FastEmbedEmbeddings()

embedding_model = load_embeddings()

# ---------- TEXT SPLITTER (FIXED) ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,   # 🔥 SMALL chunks = better fact retrieval
    chunk_overlap=30
)

# ---------- PROMPT (IMPROVED) ----------
prompt = PromptTemplate.from_template(
    """You are a helpful assistant.

Extract the exact answer from the context.
The wording in the question may differ from the document.

If answer exists, return only the answer.
If not, say "Not in document".

Question: {input}
Context: {context}

Answer:"""
)

# ---------- FUNCTIONS ----------

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()


def store_docs(docs):
    chunks = text_splitter.split_documents(docs)

    vectordb = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding_model
    )

    vectordb.add_documents(chunks)  # ✅ append, not overwrite
    vectordb.persist()

    return len(chunks)


# ---------- VECTOR DB ----------
@st.cache_resource
def get_vectorstore():
    return Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embedding_model
    )


# ---------- CHAIN ----------
@st.cache_resource
def get_chain():
    vectordb = get_vectorstore()

    retriever = vectordb.as_retriever(
        search_type="mmr",   # 🔥 better retrieval
        search_kwargs={
            "k": 5,
            "fetch_k": 10
        }
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    return chain


# ---------- UI ----------
st.set_page_config(page_title="AI Document Analyzer", layout="wide")

st.title("📄 AI Document Analyzer")

# ---------- SIDEBAR ----------
st.sidebar.header("Upload PDF")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        filepath = os.path.join(UPLOAD_FOLDER, file.name)

        with open(filepath, "wb") as f:
            f.write(file.read())

        with st.spinner(f"Processing {file.name}..."):
            docs = load_pdf(filepath)
            chunks = store_docs(docs)

        st.sidebar.success(f"{file.name} → {chunks} chunks")

    # 🔥 clear cache after new docs
    st.cache_resource.clear()


# ---------- QUERY ----------
st.subheader("Ask Questions")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your query")

if st.button("Submit") and query:
    with st.spinner("Thinking..."):

        try:
            chain = get_chain()

            # 🔥 DEBUG (optional - remove later)
            docs = get_vectorstore().similarity_search(query, k=3)
            st.write("🔍 Retrieved:", docs)

            result = chain.invoke({"input": query})

            answer = result.get("answer", "").strip()

            if not answer:
                answer = "⚠️ No relevant answer found in documents."

        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))


# ---------- CHAT DISPLAY ----------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")