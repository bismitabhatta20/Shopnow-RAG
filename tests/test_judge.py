"""
Offline tests for the LLM judge: one test per path in verify_answer().
The judge is replaced by a fake that returns prepared verdicts, so no OpenAI call is made.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app import ui
from app.judge import NOT_FOUND, Verdict, create_judge, is_approved, verify_answer
from app.models import get_judge_llm
from app.prompts import create_qa_prompt
from app.rag_chain import create_qa_chain
from test_pipeline import STORES

GOOD = Verdict(context_relevant=True, answer_relevant=True, grounded=True, score=5, reason="Correct and supported.")
INVENTED = Verdict(context_relevant=True, answer_relevant=True, grounded=False, score=2, reason="The 60 days is not in the context.")
NO_CONTEXT = Verdict(context_relevant=False, answer_relevant=True, grounded=False, score=1, reason="Context is about shipping, not returns.")


def fake_judge(*verdicts):
    """A judge that returns the given verdicts one after the other."""
    queue = list(verdicts)
    return RunnableLambda(lambda inputs: queue.pop(0))


def rewrite_chain(text):
    """A qa_chain whose rewrite always returns `text`."""
    return create_qa_chain(FakeListChatModel(responses=[text]), create_qa_prompt())


def test_is_approved_needs_all_checks_and_score():
    assert is_approved(GOOD)
    assert not is_approved(INVENTED)
    assert not is_approved(GOOD.model_copy(update={"score": 3}))


def test_good_answer_is_sent_unchanged():
    answer, verdict, rewritten = verify_answer("q", "ctx", "30 days [1].", fake_judge(GOOD), rewrite_chain("unused"))
    assert answer == "30 days [1]." and verdict.score == 5 and not rewritten


def test_irrelevant_context_gives_safe_message_without_rewrite():
    answer, verdict, rewritten = verify_answer("q", "ctx", "60 days.", fake_judge(NO_CONTEXT), rewrite_chain("unused"))
    assert answer == NOT_FOUND and not rewritten


def test_invented_fact_is_rewritten_and_rechecked():
    answer, verdict, rewritten = verify_answer("q", "ctx", "60 days.", fake_judge(INVENTED, GOOD), rewrite_chain("30 days [1]."))
    assert answer == "30 days [1]." and rewritten and verdict.score == 5


def test_rewrite_that_still_fails_is_never_sent():
    answer, verdict, rewritten = verify_answer("q", "ctx", "60 days.", fake_judge(INVENTED, INVENTED), rewrite_chain("90 days."))
    assert answer == NOT_FOUND and rewritten


def test_judge_chain_builds_with_openai_model():
    create_judge(get_judge_llm())  # builds the prompt | model chain; no request is sent


def test_ui_runs_the_judge(monkeypatch):
    ui.vector_stores.update(STORES)
    monkeypatch.setattr(ui, "get_llm", lambda: FakeListChatModel(responses=["Shipping costs 60 dollars.", "Shipping is free over $50 [1]."]))
    monkeypatch.setattr(ui, "create_judge", lambda llm: fake_judge(INVENTED, GOOD))
    answer, sources, _, verdict = ui.answer_question("How much is shipping?", "US", "Simple RAG chain")
    assert answer == "Shipping is free over $50 [1]."
    assert verdict.startswith("Approved") and "rewritten once" in verdict
