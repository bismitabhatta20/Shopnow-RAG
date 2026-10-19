"""
Step 13c - Adaptive RAG: choose a query strategy, answer, let the judge check, and switch
strategy only if the answer is rejected.

    choose_first_strategy  (free checks)
    repeat (max MAX_ATTEMPTS):
        build_queries -> retrieve_for_queries -> format_context -> qa_chain (answer) -> judge
        approved?  -> return the answer
        otherwise  -> next_strategy (from the judge's verdict)
    nothing approved -> safe "contact support" message
"""
from app.config import MAX_ATTEMPTS
from app.debug_logger import get_logger, log_block, log_calls
from app.judge import NOT_FOUND, is_approved, judge_answer
from app.query_strategies import build_queries, retrieve_for_queries
from app.retriever import format_context
from app.strategy_selector import choose_first_strategy, next_strategy

log = get_logger(__name__)


@log_calls
def answer_adaptively(question, vector_stores, qa_chain, judge, expander, decomposer, region="US",
                      max_attempts=MAX_ATTEMPTS):
    """
    Answer a question, escalating the query strategy only when the judge rejects the answer.

    Args:
        question (str): The customer's question.
        vector_stores (dict): {"knowledge": FAISS, "legal": FAISS}
        qa_chain: From create_qa_chain(); writes the answer.
        judge: From create_judge(); reviews the answer.
        expander: From create_expander().
        decomposer: From create_decomposer().
        region (str): The customer's region.
        max_attempts (int): How many strategies may be tried.

    Returns:
        tuple: (answer, attempts, last Verdict, documents used)
               attempts = one dict per try: strategy, reason, queries, score, approved
    """
    # Step 1: pick the first strategy with free checks (no LLM call)
    log.info(f"ADAPTIVE STEP 1: choose the first strategy with free checks (region={region}, max_attempts={max_attempts})")
    strategy, reason = choose_first_strategy(question, vector_stores, region)
    log.info(f"ADAPTIVE first strategy = {strategy!r} because {reason}")
    attempts, feedback = [], ""

    for attempt_number in range(1, max_attempts + 1):
        log.info("=" * 80)
        log.info(f"ADAPTIVE ATTEMPT {attempt_number}/{max_attempts}: strategy={strategy!r} reason={reason!r}")
        log.info("=" * 80)

        # Step 2: queries -> chunks -> context
        log.info(f"ATTEMPT {attempt_number} STEP 2a: build search queries with {strategy!r}")
        queries = build_queries(question, strategy, expander, decomposer)
        log.info(f"ATTEMPT {attempt_number} STEP 2b: retrieve chunks for {len(queries)} quer(ies) from both bases")
        documents = retrieve_for_queries(queries, vector_stores, region)
        log.info(f"ATTEMPT {attempt_number} STEP 2c: build the context from {len(documents)} unique chunk(s)")
        context = format_context(documents)

        # Step 3: write the answer (feedback is empty on the first try)
        log.info(f"ATTEMPT {attempt_number} STEP 3: write the answer with the QA chain "
                 f"({'with judge feedback' if feedback else 'no feedback'})")
        if feedback:
            log_block(log, "feedback added to the QA prompt", feedback, level=20)
        answer = qa_chain.invoke({"region": region, "context": context, "question": question, "feedback": feedback})
        log_block(log, f"ATTEMPT {attempt_number} answer written by the LLM", answer, level=20)

        # Step 4: let the judge check it
        log.info(f"ATTEMPT {attempt_number} STEP 4: the judge checks the answer")
        verdict = judge_answer(question, context, answer, judge)
        approved = is_approved(verdict)
        attempts.append({"strategy": strategy, "reason": reason, "queries": queries,
                         "score": verdict.score, "approved": approved})
        if approved:
            log.info(f"ADAPTIVE RESULT: attempt {attempt_number} APPROVED with strategy {strategy!r} "
                     f"(score {verdict.score}) -> send this answer")
            return answer, attempts, verdict, documents

        # Step 5: rejected -> choose another strategy based on what the judge found
        log.info(f"ATTEMPT {attempt_number} STEP 5: answer REJECTED (score {verdict.score}) -> choose another strategy")
        strategy = next_strategy(verdict, [a["strategy"] for a in attempts])
        if strategy is None:
            log.info("ADAPTIVE: no strategy left to try -> stop")
            break
        reason = f"judge: {verdict.reason}"
        feedback = f"A reviewer rejected a previous answer: {verdict.reason} Answer using only the context."

    # Step 6: no approved answer -> never send an unchecked one
    log.info(f"ADAPTIVE RESULT: no approved answer after {len(attempts)} attempt(s) -> send the safe NOT_FOUND message")
    return NOT_FOUND, attempts, verdict, documents


@log_calls
def format_attempts(attempts):
    """
    Make the attempts readable for the UI.

    Args:
        attempts (list[dict]): From answer_adaptively().

    Returns:
        str: One line per attempt, e.g. "1. direct (clear question...) -> score 5 ✔"
    """
    lines = []
    for number, attempt in enumerate(attempts, start=1):
        mark = "✔" if attempt["approved"] else "✘"
        lines.append(f"{number}. {attempt['strategy']} ({attempt['reason']}) -> "
                     f"{len(attempt['queries'])} queries, score {attempt['score']} {mark}")
    return "\n".join(lines)
