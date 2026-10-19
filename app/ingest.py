"""
The ingestion pipeline: load -> clean -> chunk -> embed -> index -> save.

Run it once (and again whenever the documents change):
    python -m app.ingest
"""
from app.chunker import chunk_documents
from app.config import KNOWLEDGE_DIR, LEGAL_DIR
from app.debug_logger import STATS, end_trace, get_log_file, get_logger, log_calls, start_trace
from app.loader import load_documents
from app.models import get_embedding_model
from app.vector_store import create_vector_store, save_vector_store

log = get_logger(__name__)


@log_calls
def build_index(folder, name, embedding_model):
    """
    Build and save the index for one document base.

    Args:
        folder (str): Folder with the Markdown files.
        name (str): "knowledge" or "legal".
        embedding_model: The embedding model.

    Returns:
        FAISS: The new vector store.
    """
    # Step 1: load the documents
    log.info(f"[{name}] STEP 1/4 load documents from {folder!r}")
    documents = load_documents(folder, name)

    # Step 2: clean + chunk them
    log.info(f"[{name}] STEP 2/4 clean + chunk {len(documents)} document(s)")
    chunks = chunk_documents(documents)

    # Step 3: embed the chunks and build the index
    log.info(f"[{name}] STEP 3/4 embed {len(chunks)} chunk(s) + build FAISS index")
    store = create_vector_store(chunks, embedding_model)

    # Step 4: save the index to disk
    log.info(f"[{name}] STEP 4/4 save index")
    save_vector_store(store, name)
    print(f"{name}: {len(documents)} documents -> {len(chunks)} chunks")
    return store


@log_calls
def build_all_indexes(embedding_model=None):
    """
    Build both indexes (knowledge base and legal base).

    Args:
        embedding_model: Optional; the OpenAI embedding model is used when not given.

    Returns:
        dict: {"knowledge": FAISS, "legal": FAISS}
    """
    STATS["ingestions"] += 1
    embedding_model = embedding_model or get_embedding_model()
    return {
        "knowledge": build_index(KNOWLEDGE_DIR, "knowledge", embedding_model),
        "legal": build_index(LEGAL_DIR, "legal", embedding_model),
    }


if __name__ == "__main__":
    start_trace("INGEST", "python -m app.ingest")
    build_all_indexes()
    end_trace("indexes built")
    print(f"Debug log: {get_log_file()}")
