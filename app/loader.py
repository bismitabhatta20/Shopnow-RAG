"""
Step 2 - Load the documents.

Each document is a Markdown file that starts with a small YAML header:

    ---
    title: EU Consumer Rights
    region: EU          # global, US, EU or UK
    ---
    # Text of the document ...
"""
from pathlib import Path

import yaml
from langchain_core.documents import Document

from app.debug_logger import get_logger, log_block, log_calls

log = get_logger(__name__)


@log_calls
def read_markdown(path):
    """
    Read one Markdown file and split it into its header (metadata) and its text.

    Args:
        path (Path): Path of the .md file.

    Returns:
        tuple: (metadata dict, body text)
    """
    text = path.read_text(encoding="utf-8")
    log.info(f"read file {str(path)!r}: {len(text)} chars")

    # The header sits between the first two '---' lines
    _, header, body = text.split("---", 2)
    metadata = yaml.safe_load(header)
    log.info(f"YAML header parsed: {metadata}")
    log_block(log, f"body of {path.name}", body)
    return metadata, body


@log_calls
def load_documents(folder, base):
    """
    Load every Markdown file in a folder (and its sub-folders) as a LangChain Document.

    Args:
        folder (str): Folder with .md files, e.g. "data/legal_base".
        base (str): Name of the document base: "knowledge" or "legal".

    Returns:
        list[Document]: One Document per file, with title, region, base and source in its metadata.
    """
    paths = sorted(Path(folder).rglob("*.md"))
    log.info(f"found {len(paths)} Markdown file(s) in {folder!r} for base {base!r}")

    documents = []
    for path in paths:
        metadata, body = read_markdown(path)
        documents.append(Document(
            page_content=body,
            metadata={
                "title": metadata.get("title", path.stem),
                "region": metadata.get("region", "global"),
                "base": base,
                "source": str(path),
            },
        ))
        log.info(f"document created: {documents[-1].metadata}")
    log.info(f"loaded {len(documents)} document(s) from {folder!r}")
    return documents
