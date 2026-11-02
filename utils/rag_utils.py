import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploaded_pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "agri_research_papers"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_PROMPT = ChatPromptTemplate.from_template(
    """You are an agricultural research assistant. Answer the user's question
using ONLY the research context provided below. Be precise, cite specific
findings where possible, and clearly say if the context does not contain
enough information to answer.

<context>
{context}
</context>

Question: {input}

Answer in a clear, well-structured way for a researcher or student."""
)


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
_embeddings = None
_vectorstore = None
_llm = None


def get_embeddings():
    """Return a cached HuggingFace sentence-transformer embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore():
    """Return a cached, persisted Chroma vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _vectorstore


def get_llm():
    """Return a cached ChatGroq client running Llama 3.3 70B."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Add it to your .env file."
            )
        _llm = ChatGroq(
            model=LLM_MODEL_NAME,
            api_key=api_key,
            temperature=0.2,
        )
    return _llm


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------
def save_uploaded_pdf(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile object to data/uploaded_pdfs/."""
    dest = UPLOAD_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def process_pdf(pdf_path: Path, source_name: str | None = None) -> int:
    """
    Load a PDF, split it into chunks, embed the chunks, and add them to
    ChromaDB. Returns the number of chunks added.
    """
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    display_name = source_name or pdf_path.name
    for chunk in chunks:
        chunk.metadata["source"] = display_name

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def process_uploaded_pdfs(uploaded_files) -> dict:
    """
    Save + process a list of Streamlit UploadedFile objects.
    Returns a summary dict: {filename: chunk_count}.
    """
    summary = {}
    for uploaded_file in uploaded_files:
        path = save_uploaded_pdf(uploaded_file)
        chunk_count = process_pdf(path, source_name=uploaded_file.name)
        summary[uploaded_file.name] = chunk_count
    return summary


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def ask_question(question: str, k: int = 4) -> dict:
    """
    Run the full RAG pipeline for a single question.
    Returns {"answer": str, "sources": [{"source": str, "page": int, "text": str}, ...]}.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = get_llm()
    document_chain = create_stuff_documents_chain(llm, RESEARCH_PROMPT)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    result = retrieval_chain.invoke({"input": question})

    sources = []
    for doc in result.get("context", []):
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "text": doc.page_content[:400],
            }
        )

    return {"answer": result["answer"], "sources": sources}


def get_document_count() -> int:
    """Return the number of chunks currently stored in ChromaDB."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def list_ingested_sources() -> list[str]:
    """Return the unique list of source filenames stored in ChromaDB."""
    try:
        data = get_vectorstore()._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in data.get("metadatas", [])}
        return sorted(sources)
    except Exception:
        return []
def ask_llm_directly(question: str) -> str:
    """
    Ask Llama 3.3 70B directly, with no ChromaDB retrieval — general
    agricultural knowledge from the model itself, not your uploaded papers.
    """
    llm = get_llm()
    response = llm.invoke(
        f"You are an agricultural expert. Answer this question using your "
        f"general knowledge of agriculture and crop science:\n\n{question}"
    )
    return response.content    
# ---------------------------------------------------------------------------
# Direct web search (bypasses ChromaDB and the model's own knowledge)
# ---------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a live web search via DuckDuckGo. Returns a list of
    {title, body, href} dicts."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def ask_with_web_search(question: str) -> dict:
    """
    Search the web for the question, then have Llama 3.3 70B summarize
    the results into an answer. Returns {"answer": str, "results": [...]}.
    """
    results = web_search(question)

    if not results:
        return {"answer": "No web results found for that query.", "results": []}

    context = "\n\n".join(
        f"Title: {r.get('title', '')}\nSnippet: {r.get('body', '')}\nSource: {r.get('href', '')}"
        for r in results
    )

    llm = get_llm()
    prompt = (
        "Answer the question using ONLY the web search results below. "
        "Mention which source(s) support your answer.\n\n"
        f"<web_results>\n{context}\n</web_results>\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content, "results": results}