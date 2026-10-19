"""
Offline tests: every step of the pipeline, with fake models (no OpenAI key, no cost).

Run:  pytest
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")  # the real models are never called in these tests

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import ui
from app.agent import create_tools, run_agent
from app.chunker import chunk_documents
from app.config import KNOWLEDGE_DIR, LEGAL_DIR
from app.loader import load_documents
from app.preprocess import remove_card_numbers
from app.prompts import create_agent_prompt, create_qa_prompt
from app.rag_chain import create_qa_chain, generate_answer
from app.retriever import retrieve
from app.vector_store import create_vector_store

WORDS = "return refund days ship cancel warranty gift card privacy data legal rights withdrawal eu uk us arbitration headphones watch".split()


class WordCountEmbeddings(Embeddings):
    """Fake embedding model: counts a few key words, so similar texts get similar vectors."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        text = text.lower()
        return [text.count(w) + 0.01 for w in WORDS]


class FakeToolModel(GenericFakeChatModel):
    """Fake chat model that returns prepared messages and accepts bind_tools()."""

    def bind_tools(self, tools, **kwargs):
        return self


def make_stores():
    """Build both indexes with the fake embedding model."""
    embeddings = WordCountEmbeddings()
    return {
        "knowledge": create_vector_store(chunk_documents(load_documents(KNOWLEDGE_DIR, "knowledge")), embeddings),
        "legal": create_vector_store(chunk_documents(load_documents(LEGAL_DIR, "legal")), embeddings),
    }


STORES = make_stores()


def test_load_documents():
    docs = load_documents(LEGAL_DIR, "legal")
    assert len(docs) == 5
    assert {d.metadata["region"] for d in docs} == {"global", "EU", "UK", "US"}


def test_chunks_are_small_and_keep_metadata():
    chunks = chunk_documents(load_documents(KNOWLEDGE_DIR, "knowledge"))
    assert len(chunks) > 20
    assert all(c.metadata["title"] and c.page_content.startswith(c.metadata["title"]) for c in chunks)


def test_card_numbers_are_removed():
    assert "4111" not in remove_card_numbers("Refund to 4111 1111 1111 1111 please")


def test_retrieve_respects_region():
    us = retrieve("withdrawal rights eu", STORES["legal"], region="US")
    eu = retrieve("withdrawal rights eu", STORES["legal"], region="EU")
    assert all(d.metadata["region"] in ("global", "US") for d in us)
    assert any(d.metadata["region"] == "EU" for d in eu)


def test_rag_chain_sends_context_and_question():
    llm = FakeListChatModel(responses=["You have 30 days [1]."])
    answer, documents = generate_answer("How many days to return?", STORES, create_qa_chain(llm, create_qa_prompt()))
    assert answer == "You have 30 days [1]." and documents


def test_agent_calls_tool_then_answers():
    llm = FakeToolModel(messages=iter([
        AIMessage("", tool_calls=[{"name": "search_knowledge_base", "args": {"query": "return days"}, "id": "1"}]),
        AIMessage("You have 30 days [1]."),
    ]))
    sources = []
    answer, steps = run_agent("How long to return?", llm, create_tools(STORES, "US", sources), create_agent_prompt())
    assert answer == "You have 30 days [1]."
    assert steps[0].startswith("search_knowledge_base") and sources


def test_date_tool():
    calculate_date = create_tools(STORES, "US", [])[2]
    assert calculate_date.invoke({"start_date": "2026-10-10", "days": 30}) == "2026-11-09 (Monday)"


def test_agent_stops_after_max_steps():
    call = AIMessage("", tool_calls=[{"name": "search_legal_base", "args": {"query": "rights"}, "id": "1"}])
    llm = FakeToolModel(messages=iter([call] * 3))
    answer, steps = run_agent("rights?", llm, create_tools(STORES, "US", []), create_agent_prompt(), max_steps=3)
    assert len(steps) == 3 and answer.startswith("Sorry")


def test_ui_answer_without_judge(monkeypatch):
    ui.vector_stores.update(STORES)
    monkeypatch.setattr(ui, "get_llm", lambda: FakeListChatModel(responses=["Shipping is free over $50 [1]."]))
    answer, sources, _, verdict = ui.answer_question("How much is shipping?", "US", "Simple RAG chain", use_judge=False)
    assert answer.startswith("Shipping") and sources and verdict == "Judge turned off."
