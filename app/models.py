"""
Step 1 - The two models.

    LLM             : reads the question + context and writes the answer (OpenAI chat model)
    Embedding model : turns text into a list of numbers (a vector) so we can search by meaning

Debug logging: the prompts sent to these models and the text they generate are logged by
debug_logger.LLMDebugCallback (attached to each chain with debug_config(...)), and every embedding
call is logged by debug_logger.LoggingEmbeddings.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import EMBEDDING_MODEL, FAST_MODEL, JUDGE_MODEL, LLM_MODEL
from app.debug_logger import get_logger, log_calls, wrap_embeddings

log = get_logger(__name__)


@log_calls
def get_llm():
    """
    Create the OpenAI chat model.

    Returns:
        ChatOpenAI: The language model. It reads OPENAI_API_KEY from the environment.
    """
    # temperature=0 -> the same question gives (almost) the same answer
    log.info(f"creating answer LLM: model={LLM_MODEL} temperature=0")
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


@log_calls
def get_judge_llm():
    """
    Create the OpenAI chat model used as the judge (it reviews answers, it does not write them).

    Returns:
        ChatOpenAI: The judge model. Set JUDGE_MODEL in .env to use a different model.
    """
    log.info(f"creating judge LLM: model={JUDGE_MODEL} temperature=0")
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


@log_calls
def get_fast_llm():
    """
    Create a cheaper, faster OpenAI chat model for small jobs (writing search queries).

    Returns:
        ChatOpenAI: The fast model. Set FAST_MODEL in .env to change it.
    """
    log.info(f"creating fast LLM: model={FAST_MODEL} temperature=0")
    return ChatOpenAI(model=FAST_MODEL, temperature=0)


@log_calls
def get_embedding_model():
    """
    Create the OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: The model used to embed document chunks and user questions
                          (wrapped so every embedding call is logged).
    """
    log.info(f"creating embedding model: model={EMBEDDING_MODEL}")
    return wrap_embeddings(OpenAIEmbeddings(model=EMBEDDING_MODEL))
