"""
Step 12 - The Gradio user interface.

Start it with:
    python -m app.ui        ->  open http://localhost:7860
"""
import time

import gradio as gr

from app.adaptive import answer_adaptively, format_attempts
from app.agent import create_tools, run_agent
from app.debug_logger import STATS, end_trace, get_log_file, get_logger, log_block, log_calls, start_trace
from app.ingest import build_all_indexes
from app.judge import create_judge, format_verdict, verify_answer
from app.models import get_embedding_model, get_fast_llm, get_judge_llm, get_llm
from app.preprocess import remove_card_numbers
from app.prompts import create_agent_prompt, create_qa_prompt
from app.query_strategies import create_decomposer, create_expander
from app.rag_chain import create_qa_chain, generate_answer
from app.retriever import format_context
from app.vector_store import load_vector_store

log = get_logger(__name__)

# The two indexes, loaded once and reused for every question
vector_stores = {}


@log_calls
def get_vector_stores():
    """
    Load the saved indexes the first time they are needed.

    Returns:
        dict: {"knowledge": FAISS, "legal": FAISS}
    """
    if vector_stores:
        log.info("indexes already loaded -> reuse them")
    else:
        log.info("indexes not loaded yet -> load them from disk")
        embedding_model = get_embedding_model()
        vector_stores["knowledge"] = load_vector_store("knowledge", embedding_model)
        vector_stores["legal"] = load_vector_store("legal", embedding_model)
    return vector_stores


def build_indexes():
    """
    Run the ingestion pipeline from the UI (button "Build index").

    Returns:
        str: A status message.
    """
    start_trace("INGEST", "button 'Build index' in the UI")
    try:
        vector_stores.update(build_all_indexes())
    finally:
        end_trace("ingestion finished")
    return "Indexes built: knowledge base and legal base."


@log_calls
def format_sources(documents):
    """
    Make a short, readable list of the documents used.

    Args:
        documents (list[Document]): Retrieved chunks.

    Returns:
        str: One line per unique document title.
    """
    titles = dict.fromkeys(f"{doc.metadata['title']} ({doc.metadata['region']})" for doc in documents)
    return "\n".join(titles)


def _answer_question(question, region, mode, use_judge=True):
    """
    Answer the customer's question with the agent or the simple RAG chain, then let the judge check it.

    Args:
        question (str): The customer's question.
        region (str): "US", "EU" or "UK".
        mode (str): "Agent (tool calling)", "Simple RAG chain" or "Adaptive RAG".
        use_judge (bool): Check the answer with the LLM judge (Adaptive RAG always uses the judge).

    Returns:
        tuple: (answer, sources, agent steps, judge verdict)
    """
    if not question.strip():
        log.info("empty question -> ask the user to type one")
        return "Please type a question.", "", "", ""

    # Step 1: never send card numbers to OpenAI
    log.info("UI STEP 1: remove card numbers from the question")
    question = remove_card_numbers(question)

    # Step 2: get the models, the indexes and the answer chain (the judge also uses it to rewrite answers)
    log.info("UI STEP 2: create the models, load the indexes, build the QA chain")
    llm = get_llm()
    stores = get_vector_stores()
    qa_chain = create_qa_chain(llm, create_qa_prompt())

    # Adaptive RAG: strategy choice + judge are part of the loop, so it returns directly
    if mode == "Adaptive RAG":
        log.info("UI: mode = Adaptive RAG -> strategy choice + judge happen inside answer_adaptively()")
        fast_llm = get_fast_llm()
        answer, attempts, verdict, documents = answer_adaptively(
            question, stores, qa_chain, create_judge(get_judge_llm()),
            create_expander(fast_llm), create_decomposer(fast_llm), region)
        return answer, format_sources(documents), format_attempts(attempts), format_verdict(verdict, False)

    # Step 3: write the first answer
    log.info(f"UI STEP 3: write the first answer with mode {mode!r}")
    if mode == "Agent (tool calling)":
        documents = []                     # the search tools add every chunk they find here
        tools = create_tools(stores, region, documents)
        answer, steps = run_agent(question, llm, tools, create_agent_prompt(), region)
        steps_text = "\n".join(steps)
    else:
        answer, documents = generate_answer(question, stores, qa_chain, region)
        steps_text = "(no tools in the simple chain)"

    # Step 4: let the judge check the answer against the same context
    log.info(f"UI STEP 4: judge check (use_judge={use_judge}, {len(documents)} document(s) used)")
    if not use_judge:
        verdict_text = "Judge turned off."
    elif not documents:
        verdict_text = "Not checked: no documents were used (small talk or hand-off to a person)."
    else:
        judge = create_judge(get_judge_llm())
        answer, verdict, rewritten = verify_answer(question, format_context(documents), answer, judge, qa_chain, region)
        verdict_text = format_verdict(verdict, rewritten)

    return answer, format_sources(documents), steps_text, verdict_text


def answer_question(question, region, mode, use_judge=True):
    """
    Entry point of the "Ask" button: answers the question (see _answer_question) and writes the
    whole flow of this question to the debug log under its own trace id (Q1, Q2, ...).

    Not decorated with @log_calls on purpose: the raw question may contain a card number,
    and only the cleaned question may be written to the log.

    Returns:
        tuple: (answer, sources, agent steps, judge verdict)
    """
    STATS["questions"] += 1
    start_trace("Q", f"mode={mode!r} region={region!r} use_judge={use_judge}")
    started = time.perf_counter()
    try:
        answer, sources, steps_text, verdict_text = _answer_question(question, region, mode, use_judge)
    except Exception:
        log.exception("answer_question failed")
        end_trace(f"FAILED after {time.perf_counter() - started:.2f}s")
        raise
    log.info("UI OUTPUT shown to the customer:")
    log_block(log, "Answer", answer, level=20)
    log_block(log, "Sources used", sources, level=20)
    log_block(log, "Steps / strategies tried", steps_text, level=20)
    log_block(log, "Judge verdict", verdict_text, level=20)
    end_trace(f"answered in {time.perf_counter() - started:.2f}s")
    return answer, sources, steps_text, verdict_text


with gr.Blocks(title="ShopNow Support") as interface:

    gr.Markdown("<h2 style='text-align: center;'>ShopNow Customer Support (RAG)</h2>")

    # Inputs
    with gr.Row():
        region_input = gr.Radio(["US", "EU", "UK"], value="US", label="Customer region")
        mode_input = gr.Radio(["Agent (tool calling)", "Simple RAG chain", "Adaptive RAG"],
                              value="Adaptive RAG", label="Mode")
        judge_input = gr.Checkbox(value=True, label="Use LLM judge")
    question_input = gr.Textbox(label="Your question", placeholder="e.g. How long do I have to return headphones?")
    ask_btn = gr.Button("Ask", variant="primary")

    # Outputs
    answer_output = gr.Textbox(label="Answer", lines=8)
    sources_output = gr.Textbox(label="Sources used", lines=3)
    steps_output = gr.Textbox(label="Steps (agent tool calls / adaptive strategies tried)", lines=3)
    verdict_output = gr.Textbox(label="Judge verdict", lines=2)

    # Index management
    build_btn = gr.Button("Build index (run once, or after changing documents)")
    build_status = gr.Textbox(label="Index status", interactive=False)

    # Button actions
    ask_btn.click(answer_question, inputs=[question_input, region_input, mode_input, judge_input],
                  outputs=[answer_output, sources_output, steps_output, verdict_output])
    build_btn.click(build_indexes, outputs=build_status)


if __name__ == "__main__":
    print(f"Debug log for this session: {get_log_file()}")
    interface.launch(server_name="0.0.0.0", server_port=7860)
