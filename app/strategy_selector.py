"""
Step 13b - Choose the query strategy, and use the expensive ones ONLY when they are needed.

First choice (free: plain Python, no LLM call):
    multi-part question?   -> decomposition
    weak search results?   -> expansion
    otherwise              -> direct (no extra cost)

Next choice, only after the judge rejects an answer:
    context not relevant   -> expansion first (search missed: try other words)
    answer incomplete etc. -> decomposition first (part of the question not covered)
    both tried             -> decompose+expand
"""
from app.config import MULTI_PART_MIN_WORDS, WEAK_RETRIEVAL_DISTANCE
from app.debug_logger import get_logger, log_calls, store_label
from app.query_strategies import COMBINED, DECOMPOSITION, DIRECT, EXPANSION

log = get_logger(__name__)

# Words that usually join two separate questions
JOINERS = (" and also ", " as well as ", " also ", " plus ", " in addition ")


@log_calls
def is_multi_part(question):
    """
    Guess, without an LLM, if a question asks several things.

    Args:
        question (str): The customer's question.

    Returns:
        bool: True if it has 2+ question marks, a joining phrase, or is very long.
    """
    text = f" {question.lower()} "
    question_marks = question.count("?")
    joiners_found = [joiner.strip() for joiner in JOINERS if joiner in text]
    words = len(question.split())
    result = question_marks > 1 or bool(joiners_found) or words > MULTI_PART_MIN_WORDS
    log.info(f"multi-part check: question marks={question_marks} (>1? {question_marks > 1}), "
             f"joining phrases={joiners_found}, words={words} (>{MULTI_PART_MIN_WORDS}? "
             f"{words > MULTI_PART_MIN_WORDS}) -> multi-part={result}")
    return result


@log_calls
def best_distance(question, vector_stores, region):
    """
    Distance between the question and the closest chunk in either base (no LLM; only the question is embedded).

    Args:
        question (str): The customer's question.
        vector_stores (dict): {"knowledge": FAISS, "legal": FAISS}
        region (str): The customer's region.

    Returns:
        float: The smallest distance found. Smaller = better match.
    """
    distances = []
    for store in vector_stores.values():
        results = store.similarity_search_with_score(
            question, k=1, filter=lambda metadata: metadata["region"] in ("global", region), fetch_k=20)
        for doc, score in results:
            log.info(f"closest chunk in {store_label(store)}: distance={float(score):.4f} "
                     f"title={doc.metadata.get('title')!r} section={doc.metadata.get('section', '')!r}")
        distances += [score for _, score in results]
    best = min(distances)
    log.info(f"best distance = {float(best):.4f} (weak if > WEAK_RETRIEVAL_DISTANCE={WEAK_RETRIEVAL_DISTANCE})")
    return best


@log_calls
def choose_first_strategy(question, vector_stores, region):
    """
    Pick the cheapest strategy that should work, before any answer is written.

    Args:
        question (str): The customer's question.
        vector_stores (dict): {"knowledge": FAISS, "legal": FAISS}
        region (str): The customer's region.

    Returns:
        tuple: (strategy, reason)
    """
    log.info("FREE CHECK 1/2: is the question multi-part?")
    if is_multi_part(question):
        log.info(f"🎯 STRATEGY SELECTED: {DECOMPOSITION!r} (question has several parts)")
        return DECOMPOSITION, "question has several parts"

    log.info("FREE CHECK 2/2: how good is the best search result?")
    distance = best_distance(question, vector_stores, region)
    if distance > WEAK_RETRIEVAL_DISTANCE:
        log.info(f"🎯 STRATEGY SELECTED: {EXPANSION!r} (distance {distance:.2f} > {WEAK_RETRIEVAL_DISTANCE})")
        return EXPANSION, f"weak search results (best distance {distance:.2f})"

    log.info(f"🎯 STRATEGY SELECTED: {DIRECT!r} (distance {distance:.2f} <= {WEAK_RETRIEVAL_DISTANCE})")
    return DIRECT, f"clear question, good search results (best distance {distance:.2f})"


@log_calls
def next_strategy(verdict, tried):
    """
    After the judge rejected an answer, pick a strategy that has not been tried yet.

    Args:
        verdict (Verdict): The judge's review of the last answer.
        tried (list[str]): Strategies already used for this question.

    Returns:
        str or None: The next strategy, or None if every useful one was tried.
    """
    if not verdict.context_relevant:
        order = [EXPANSION, DECOMPOSITION, COMBINED]   # the search missed -> try other words first
        why = "context NOT relevant (the search missed) -> try other words first"
    else:
        order = [DECOMPOSITION, EXPANSION, COMBINED]   # part of the question not covered -> split it first
        why = "context relevant but answer rejected (part not covered?) -> split the question first"
    log.info(f"escalation rule: {why}; preference order {order}; already tried {tried}")

    for strategy in order:
        if strategy not in tried:
            log.info(f"🎯 NEXT STRATEGY SELECTED: {strategy!r}")
            return strategy
    log.info("🎯 NEXT STRATEGY: none left (every strategy was tried)")
    return None
