"""
Step 9 - Generation with a simple RAG chain (fixed steps, no tools):

    question -> retrieve chunks -> build context -> prompt (context + question) -> LLM -> answer
"""
from langchain_core.output_parsers import StrOutputParser

from app.debug_logger import debug_config, get_logger, log_block, log_calls
from app.retriever import format_context, retrieve

log = get_logger(__name__)


@log_calls
def create_qa_chain(llm, prompt):
    """
    Chain the prompt, the LLM and an output parser together.

    Args:
        llm: The chat model.
        prompt (ChatPromptTemplate): From create_qa_prompt().

    Returns:
        Runnable: Call .invoke({...}) with region, context, question and feedback to get the answer text.
    """
    # prompt fills in the text -> llm writes the answer -> parser keeps only the answer text
    # .with_config(debug_config(...)) logs every LLM call this chain makes (full prompt + answer)
    log.info(f"QA chain created: prompt -> {type(llm).__name__} -> StrOutputParser")
    return (prompt | llm | StrOutputParser()).with_config(debug_config("qa_chain (answer writer)"))


@log_calls
def generate_answer(question, vector_stores, qa_chain, region="US"):
    """
    Retrieve context from both bases and generate the answer.

    Args:
        question (str): The customer's question.
        vector_stores (dict): {"knowledge": FAISS, "legal": FAISS}
        qa_chain: From create_qa_chain().
        region (str): The customer's region.

    Returns:
        tuple: (answer text, list of the chunks used as context)
    """
    # Step 1: retrieve relevant chunks from both bases
    log.info("SIMPLE RAG STEP 1/3 retrieve chunks from the knowledge base and the legal base")
    documents = retrieve(question, vector_stores["knowledge"], region) + retrieve(question, vector_stores["legal"], region)

    # Step 2: turn the chunks into one context text
    log.info(f"SIMPLE RAG STEP 2/3 build the context from {len(documents)} chunk(s)")
    context = format_context(documents)

    # Step 3: send context + question to the LLM (this is the OpenAI call)
    log.info("SIMPLE RAG STEP 3/3 call the LLM with context + question")
    answer = qa_chain.invoke({"region": region, "context": context, "question": question, "feedback": ""})
    log_block(log, "SIMPLE RAG answer", answer, level=20)
    return answer, documents
