"""
Step 4 - Chunking.

Strategy:
    1. Split each document at its headings (#, ##), so one chunk = one topic.
    2. Split long sections again into pieces of at most CHUNK_SIZE characters, with some overlap
       so a sentence that falls on a border is not lost.
    3. Put the document title in front of every chunk, so a small chunk still says where it comes from.
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.debug_logger import get_logger, log_calls, log_documents
from app.preprocess import clean_text

log = get_logger(__name__)


@log_calls
def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents into small chunks ready to be embedded.

    Args:
        documents (list[Document]): Documents from load_documents().
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Characters shared by two neighbouring chunks.

    Returns:
        list[Document]: The chunks. Each keeps the metadata of its document.
    """
    log.info(f"chunking {len(documents)} document(s): chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
             f"split at headings # and ##")
    header_splitter = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2")])
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in documents:
        log.info(f"── document {doc.metadata['title']!r} ({len(doc.page_content)} chars)")

        # 1. split by headings
        sections = header_splitter.split_text(clean_text(doc.page_content))
        log.info(f"   step 1 heading split -> {len(sections)} section(s): "
                 f"{[s.metadata.get('h2') or s.metadata.get('h1', '') for s in sections]}")

        # 2. split long sections by size
        pieces = size_splitter.split_documents(sections)
        log.info(f"   step 2 size split -> {len(pieces)} piece(s), sizes {[len(p.page_content) for p in pieces]}")

        # 3. add the title in front and keep the document metadata
        new_chunks = []
        for piece in pieces:
            piece.page_content = f"{doc.metadata['title']}\n{piece.page_content}"
            piece.metadata = {**doc.metadata, "section": piece.metadata.get("h2", "")}
            new_chunks.append(piece)
        log_documents(log, f"   step 3 chunks created for {doc.metadata['title']!r}", new_chunks)
        chunks.extend(new_chunks)

    log.info(f"total chunks: {len(chunks)}")
    return chunks
