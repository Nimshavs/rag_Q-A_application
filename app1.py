import os
import shutil
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
        print("🗑️ Old DB removed")

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

    print("\n📄 Loading PDFs...")
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"➡️ {file}")

            loader = PDFPlumberLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata["page"] = d.metadata.get("page", 0)

            docs_all.extend(docs)

    if not docs_all:
        print("❌ No PDFs found")
        return

    print("\n🔪 Splitting into chunks...")
    chunks = splitter.split_documents(docs_all)

    print(f"✅ Total chunks: {len(chunks)}")

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    vectordb.add_documents(chunks)
    vectordb.persist()

    print("💾 DB ready")


# ---------------- ASK ----------------
def ask(query):
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding
    )

    print("\n❓ Question:", query)

    # 🔥 retrieve multiple for better answer
    docs = vectordb.similarity_search(query, k=5)

    print("\n🔍 Retrieved Chunks:")
    context_texts = []

    for i, d in enumerate(docs):
        print(f"\n--- Chunk {i+1} (page {d.metadata.get('page')}) ---")
        print(d.page_content[:200])

        context_texts.append(d.page_content.strip())

    # 🧠 Build context
    context = "\n\n".join(context_texts)

    final_prompt = prompt.format(
        context=context,
        question=query
    )

    print("\n🧠 Prompt sent to LLM:\n", final_prompt[:500])

    # 🤖 LLM call
    response = llm.invoke(final_prompt)
    answer = response.content.strip()

    # ✅ ONLY take most relevant page
    best_page = docs[0].metadata.get("page", "N/A")

    return f"{answer}\n\n📄 Page: {best_page}"


# ---------------- RUN ----------------
if __name__ == "__main__":
    PDF_FOLDER = "uploads"

    reset_db()
    ingest(PDF_FOLDER)

    while True:
        q = input("\nAsk: ")
        print("\n✅ Answer:\n", ask(q))