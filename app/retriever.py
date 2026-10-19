"""
Step 7 - Retrieval.

Find the chunks that best match the question, and turn them into one block of text (the "context")
that we give to the LLM together with the question.
"""
from app.config import TOP_K
from app.debug_logger import get_logger, log_block, log_calls, log_documents, store_label

log = get_logger(__name__)


@log_calls
def retrieve(query, vector_store, region="US", k=TOP_K):
    """
    Return the k chunks most similar to the query, only for the customer's region (or global).

    Args:
        query (str): The search text (usually the user's question).
        vector_store (FAISS): The index to search.
        region (str): "US", "EU" or "UK". EU-only rules are never returned to a US customer.
        k (int): Number of chunks to return.

    Returns:
        list[Document]: The most relevant chunks.
    """
    log.info(f"search {store_label(vector_store)} for {query!r} (region filter: global+{region}, k={k}, fetch_k=20)")
    # similarity_search_with_score returns the same chunks as similarity_search, plus the distances
    # (logged for debugging; smaller distance = closer match)
    results = vector_store.similarity_search_with_score(
        query,
        k=k,
        filter=lambda metadata: metadata["region"] in ("global", region),
        fetch_k=20,  # look at 20 candidates, keep the k best that pass the filter
    )
    documents = [doc for doc, _ in results]
    log_documents(log, f"retrieved from {store_label(vector_store)}", documents, [float(s) for _, s in results])
    return documents


@log_calls
def format_context(documents):
    """
    Join the retrieved chunks into one text, numbering each source so the LLM can cite it.

    Args:
        documents (list[Document]): Chunks from retrieve().

    Returns:
        str: The context, e.g. "[1] (Returns and Refunds Policy) Most items can be returned ..."
    """
    context = "\n\n".join(
        f"[{i}] ({doc.metadata['title']}) {doc.page_content}"
        for i, doc in enumerate(documents, start=1)
    )
    log_block(log, f"CONTEXT built from {len(documents)} chunk(s)", context)
    return context
