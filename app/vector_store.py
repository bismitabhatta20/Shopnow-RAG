"""
Step 5 and 6 - Embeddings and indexing.

FAISS stores one vector per chunk and finds the vectors closest to a question's vector.
We keep two indexes: one for the knowledge base and one for the legal base.
"""
import os

from langchain_community.vectorstores import FAISS

from app.config import INDEX_DIR
from app.debug_logger import get_logger, log_calls, store_label, wrap_embeddings

log = get_logger(__name__)


@log_calls
def create_vector_store(chunks, embedding_model):
    """
    Embed the chunks and build a FAISS index from them.

    Args:
        chunks (list[Document]): Chunks from chunk_documents().
        embedding_model: The embedding model (calls OpenAI once per batch of chunks).

    Returns:
        FAISS: The vector store (index + the chunk texts and metadata).
    """
    log.info(f"embedding {len(chunks)} chunk(s) and building a FAISS index")
    store = FAISS.from_documents(chunks, wrap_embeddings(embedding_model))
    log.info(f"FAISS index built: {store_label(store)}, vector size {store.index.d}")
    return store


@log_calls
def save_vector_store(store, name):
    """
    Save an index to disk so we do not need to embed the documents again.

    Args:
        store (FAISS): The vector store.
        name (str): "knowledge" or "legal".
    """
    folder = os.path.join(INDEX_DIR, name)
    store.save_local(folder)
    for file_name in sorted(os.listdir(folder)):
        path = os.path.join(folder, file_name)
        log.info(f"file written: {os.path.abspath(path)} ({os.path.getsize(path)} bytes)")


@log_calls
def load_vector_store(name, embedding_model):
    """
    Load a saved index from disk.

    Args:
        name (str): "knowledge" or "legal".
        embedding_model: The same embedding model that built the index (used to embed questions).

    Returns:
        FAISS: The vector store.
    """
    folder = os.path.join(INDEX_DIR, name)
    log.info(f"loading FAISS index from {os.path.abspath(folder)}")
    # allow_dangerous_deserialization: the files were written by us (step 6), so they are safe to load
    store = FAISS.load_local(folder, wrap_embeddings(embedding_model), allow_dangerous_deserialization=True)
    log.info(f"loaded {store_label(store)}")
    return store
