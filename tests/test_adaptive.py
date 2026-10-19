"""
Offline tests for Adaptive RAG: the free checks, each strategy, the escalation rules and the cost
(expensive strategies must NOT run when they are not needed). Fake models only.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app import strategy_selector, ui
from app.adaptive import answer_adaptively
from app.judge import NOT_FOUND
from app.prompts import create_qa_prompt
from app.query_strategies import COMBINED, DECOMPOSITION, DIRECT, EXPANSION, QueryList, build_queries, retrieve_for_queries
from app.rag_chain import create_qa_chain
from app.strategy_selector import choose_first_strategy, is_multi_part, next_strategy
from test_judge import GOOD, INVENTED, NO_CONTEXT, fake_judge
from test_pipeline import STORES

CALLS = []  # records every fake LLM call that writes queries (to check the cost)


def fake_expander():
    return RunnableLambda(lambda x: CALLS.append("expand") or QueryList(queries=[x["question"] + " policy", "return window"]))


def fake_decomposer():
    return RunnableLambda(lambda x: CALLS.append("decompose") or QueryList(queries=["return window?", "refund method?"]))


def qa(*answers):
    return create_qa_chain(FakeListChatModel(responses=list(answers)), create_qa_prompt())


def test_is_multi_part():
    assert not is_multi_part("How long can I return headphones?")
    assert is_multi_part("How long can I return it? And how is the refund paid?")
    assert is_multi_part("What is the return window as well as the refund method")


def test_first_strategy_is_free_and_direct_for_a_simple_question(monkeypatch):
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", 1e9)  # every search counts as good
    assert choose_first_strategy("How many days to return?", STORES, "US")[0] == DIRECT


def test_weak_search_starts_with_expansion(monkeypatch):
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", -1)  # every search counts as weak
    assert choose_first_strategy("How many days to return?", STORES, "US")[0] == EXPANSION


def test_multi_part_starts_with_decomposition():
    assert choose_first_strategy("Return window? Refund method?", STORES, "US")[0] == DECOMPOSITION


def test_build_queries_per_strategy():
    CALLS.clear()
    e, d = fake_expander(), fake_decomposer()
    assert build_queries("q", DIRECT, e, d) == ["q"] and CALLS == []          # direct = no LLM call
    assert build_queries("q", EXPANSION, e, d) == ["q", "q policy", "return window"]
    assert build_queries("q", DECOMPOSITION, e, d) == ["return window?", "refund method?"]
    assert len(build_queries("q", COMBINED, e, d)) == 6                      # 2 sub-questions x (1 + 2 rewrites)


def test_retrieve_for_queries_removes_duplicates():
    docs = retrieve_for_queries(["return days", "return days"], STORES, "US")
    assert len(docs) == len({d.page_content for d in docs})


def test_next_strategy_rules():
    assert next_strategy(NO_CONTEXT, [DIRECT]) == EXPANSION       # search missed -> other words
    assert next_strategy(INVENTED, [DIRECT]) == DECOMPOSITION     # incomplete -> split the question
    assert next_strategy(INVENTED, [DIRECT, DECOMPOSITION, EXPANSION]) == COMBINED
    assert next_strategy(INVENTED, [DIRECT, DECOMPOSITION, EXPANSION, COMBINED]) is None


def test_good_first_answer_costs_no_extra_calls(monkeypatch):
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", 1e9)
    CALLS.clear()
    answer, attempts, verdict, docs = answer_adaptively(
        "How many days to return?", STORES, qa("30 days [1]."), fake_judge(GOOD), fake_expander(), fake_decomposer())
    assert answer == "30 days [1]." and len(attempts) == 1 and CALLS == []


def test_rejected_context_escalates_to_expansion(monkeypatch):
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", 1e9)
    CALLS.clear()
    answer, attempts, verdict, docs = answer_adaptively(
        "How many days to return?", STORES, qa("60 days.", "30 days [1]."),
        fake_judge(NO_CONTEXT, GOOD), fake_expander(), fake_decomposer())
    assert [a["strategy"] for a in attempts] == [DIRECT, EXPANSION]
    assert answer == "30 days [1]." and CALLS == ["expand"]


def test_nothing_approved_sends_safe_message(monkeypatch):
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", 1e9)
    answer, attempts, verdict, docs = answer_adaptively(
        "How many days to return?", STORES, qa("a", "b", "c"),
        fake_judge(INVENTED, INVENTED, INVENTED), fake_expander(), fake_decomposer())
    assert answer == NOT_FOUND and len(attempts) == 3


def test_ui_adaptive_mode(monkeypatch):
    ui.vector_stores.update(STORES)
    monkeypatch.setattr(strategy_selector, "WEAK_RETRIEVAL_DISTANCE", 1e9)
    monkeypatch.setattr(ui, "get_llm", lambda: FakeListChatModel(responses=["Shipping is free over $50 [1]."]))
    monkeypatch.setattr(ui, "create_judge", lambda llm: fake_judge(GOOD))
    answer, sources, steps, verdict = ui.answer_question("How much is shipping?", "US", "Adaptive RAG")
    assert answer.startswith("Shipping") and steps.startswith("1. direct") and verdict.startswith("Approved")
