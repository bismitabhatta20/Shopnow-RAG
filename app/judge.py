"""
Step 11 - LLM as a judge: check every answer before the customer sees it.

    answer -> judge -> approved?             -> send the answer
                    -> context not relevant? -> send a safe "not found" message (rewriting cannot help)
                    -> otherwise             -> rewrite once with the judge's feedback -> judge again
                                                -> approved? send it : send the safe message
"""
from pydantic import BaseModel, Field

from app.config import JUDGE_MIN_SCORE
from app.debug_logger import debug_config, get_logger, log_block, log_calls
from app.prompts import create_judge_prompt

log = get_logger(__name__)

NOT_FOUND = ("I'm sorry, I couldn't find a reliable answer to that in our documents. "
             "Please contact support@shop.example.com and a colleague will help you.")


class Verdict(BaseModel):
    """
    The judge's review of one answer.

    Pydantic (not a plain dict) because this comes from the LLM: Pydantic checks that every field is
    there with the right type, and LangChain sends this schema to OpenAI so the reply always fits it.
    """
    context_relevant: bool = Field(description="The context contains what is needed to answer the question")
    answer_relevant: bool = Field(description="The answer addresses the customer's question")
    grounded: bool = Field(description="Every fact in the answer is supported by the context")
    score: int = Field(description="Overall quality from 1 (unusable) to 5 (correct and supported)")
    reason: str = Field(description="One sentence: what is wrong, or why it is good")


@log_calls
def create_judge(llm):
    """
    Build the judge chain.

    Args:
        llm: The judge chat model (see get_judge_llm()).

    Returns:
        Runnable: Call .invoke({question, context, answer}) to get a Verdict object.
    """
    # with_structured_output: the LLM must reply with a Verdict, not free text
    log.info(f"judge chain created: judge prompt -> {type(llm).__name__}.with_structured_output(Verdict)")
    return (create_judge_prompt() | llm.with_structured_output(Verdict)).with_config(debug_config("judge"))


@log_calls
def judge_answer(question, context, answer, judge):
    """
    Ask the judge to review one answer.

    Args:
        question (str): The customer's question.
        context (str): The retrieved passages the answer should be based on.
        answer (str): The answer to review.
        judge: From create_judge().

    Returns:
        Verdict: The three checks, the score and the reason.
    """
    log_block(log, "JUDGE reviewing this answer", answer)
    verdict = judge.invoke({"question": question, "context": context, "answer": answer})
    log.info(f"⚖ JUDGE VERDICT: score={verdict.score}/5 context_relevant={verdict.context_relevant} "
             f"answer_relevant={verdict.answer_relevant} grounded={verdict.grounded} -> "
             f"{'APPROVED' if is_approved(verdict) else 'REJECTED'} (min score {JUDGE_MIN_SCORE})")
    log.info(f"⚖ JUDGE reason: {verdict.reason}")
    return verdict


@log_calls
def is_approved(verdict):
    """
    Decide if an answer may be sent to the customer.

    Args:
        verdict (Verdict): From judge_answer().

    Returns:
        bool: True only if all three checks pass and the score is high enough.
    """
    return (verdict.context_relevant and verdict.answer_relevant and verdict.grounded
            and verdict.score >= JUDGE_MIN_SCORE)


@log_calls
def verify_answer(question, context, answer, judge, qa_chain, region="US"):
    """
    Check an answer with the judge; rewrite it once if needed; never send an unapproved answer.

    Args:
        question (str): The customer's question.
        context (str): The retrieved passages (from format_context()).
        answer (str): The first answer (from the simple chain or the agent).
        judge: From create_judge().
        qa_chain: From create_qa_chain(); used to rewrite the answer.
        region (str): The customer's region.

    Returns:
        tuple: (final answer, last Verdict, True if the answer was rewritten)
    """
    # Step 1: review the first answer
    log.info("VERIFY STEP 1: judge the first answer")
    verdict = judge_answer(question, context, answer, judge)
    if is_approved(verdict):
        log.info("VERIFY decision: approved -> send the first answer unchanged")
        return answer, verdict, False

    # Step 2: the passages do not cover the question -> a rewrite would only guess
    if not verdict.context_relevant:
        log.info("VERIFY STEP 2: context not relevant -> rewriting cannot help -> send the safe NOT_FOUND message")
        return NOT_FOUND, verdict, False

    # Step 3: rewrite once, telling the LLM what the judge found wrong
    feedback = f"A reviewer rejected a previous answer: {verdict.reason} Write a corrected answer using only the context."
    log.info("VERIFY STEP 3: rejected but context is relevant -> rewrite once with the judge's feedback")
    log_block(log, "feedback added to the QA prompt", feedback, level=20)
    new_answer = qa_chain.invoke({"region": region, "context": context, "question": question, "feedback": feedback})
    log_block(log, "rewritten answer", new_answer, level=20)

    # Step 4: review the rewritten answer
    log.info("VERIFY STEP 4: judge the rewritten answer")
    verdict = judge_answer(question, context, new_answer, judge)
    if is_approved(verdict):
        log.info("VERIFY decision: rewritten answer approved -> send it")
        return new_answer, verdict, True
    log.info("VERIFY decision: rewritten answer still rejected -> send the safe NOT_FOUND message")
    return NOT_FOUND, verdict, True


@log_calls
def format_verdict(verdict, rewritten):
    """
    Make the verdict readable for the UI.

    Args:
        verdict (Verdict): The last verdict.
        rewritten (bool): Whether the answer was rewritten.

    Returns:
        str: e.g. "Approved · score 5/5 · context ✔ relevance ✔ grounded ✔ · <reason>"
    """
    mark = lambda ok: "✔" if ok else "✘"  # noqa: E731
    status = "Approved" if is_approved(verdict) else "Rejected (safe message sent)"
    return (f"{status} · score {verdict.score}/5 · context {mark(verdict.context_relevant)} "
            f"relevance {mark(verdict.answer_relevant)} grounded {mark(verdict.grounded)}"
            f"{' · rewritten once' if rewritten else ''}\nReason: {verdict.reason}")
