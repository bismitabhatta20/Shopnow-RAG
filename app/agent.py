"""
Step 10 - The agent: an LLM that can call tools, in a simple loop (no LangGraph).

    1. Send the messages + the list of tools to the LLM.
    2. If the LLM asks for tools, run them and add the results to the messages. Go back to 1.
    3. If the LLM answers with plain text, that is the final answer.
"""
from datetime import date, timedelta

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from app.config import MAX_AGENT_STEPS
from app.debug_logger import debug_config, get_logger, log_block, log_calls
from app.retriever import format_context, retrieve

log = get_logger(__name__)


@log_calls
def create_tools(vector_stores, region, sources):
    """
    Create the tools the agent can use.

    The region is fixed here (not chosen by the LLM), so the agent can only search
    documents for the customer's own region.

    Args:
        vector_stores (dict): {"knowledge": FAISS, "legal": FAISS}
        region (str): The customer's region.
        sources (list): Every chunk a search tool finds is added here, to show as sources later.

    Returns:
        list: The tools.
    """
    # The tools log by hand (not with @log_calls) because @tool reads the function's signature.

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search store policies (returns, refunds, shipping, payments, warranty, gift cards), the FAQ and product guides."""
        log.info(f"🔧 TOOL search_knowledge_base(query={query!r}) region={region}")
        documents = retrieve(query, vector_stores["knowledge"], region)
        sources.extend(documents)
        result = format_context(documents)
        log.info(f"🔧 TOOL search_knowledge_base returned {len(documents)} chunk(s), {len(result)} chars")
        return result

    @tool
    def search_legal_base(query: str) -> str:
        """Search the terms of service, privacy policy and consumer-rights documents."""
        log.info(f"🔧 TOOL search_legal_base(query={query!r}) region={region}")
        documents = retrieve(query, vector_stores["legal"], region)
        sources.extend(documents)
        result = format_context(documents)
        log.info(f"🔧 TOOL search_legal_base returned {len(documents)} chunk(s), {len(result)} chars")
        return result

    @tool
    def calculate_date(start_date: str, days: int) -> str:
        """Add a number of days to a date (format YYYY-MM-DD) and return the new date."""
        new_date = date.fromisoformat(start_date) + timedelta(days=days)
        result = new_date.strftime("%Y-%m-%d (%A)")
        log.info(f"🔧 TOOL calculate_date(start_date={start_date!r}, days={days}) -> {result!r}")
        return result

    @tool
    def escalate_to_human(summary: str) -> str:
        """Hand the conversation to a human support agent. Use only if the customer asks for a person."""
        log.info(f"🔧 TOOL escalate_to_human(summary={summary!r})")
        return "A human support agent will contact the customer by email."

    tools = [search_knowledge_base, search_legal_base, calculate_date, escalate_to_human]
    log.info(f"tools created for region {region}: {[t.name for t in tools]}")
    return tools


@log_calls
def run_agent(question, llm, tools, prompt, region="US", max_steps=MAX_AGENT_STEPS):
    """
    Run the agent loop for one question.

    Args:
        question (str): The customer's question.
        llm: The chat model (must support tool calling).
        tools (list): From create_tools().
        prompt (ChatPromptTemplate): From create_agent_prompt().
        region (str): The customer's region.
        max_steps (int): Safety limit on the number of LLM calls.

    Returns:
        tuple: (answer text, list of tool calls made, e.g. ["search_knowledge_base({'query': ...})"])
    """
    # Step 1: tell the LLM which tools exist
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    log.info(f"AGENT: LLM bound to tools {list(tools_by_name)}; max_steps={max_steps}")

    # Step 2: start the conversation: system rules + the question
    messages = prompt.format_messages(region=region, question=question)
    log.info(f"AGENT: conversation started with {len(messages)} message(s)")

    steps = []
    for step in range(1, max_steps + 1):
        # Step 3: ask the LLM what to do next
        log.info(f"AGENT LOOP step {step}/{max_steps}: asking the LLM what to do next ({len(messages)} messages so far)")
        response = llm_with_tools.invoke(messages, config=debug_config("agent", agent_step=step))
        messages.append(response)

        # Step 4: no tool requested -> this is the final answer
        if not response.tool_calls:
            log.info(f"AGENT LOOP step {step}: no tool requested -> final answer")
            log_block(log, "AGENT final answer", response.content, level=20)
            return response.content, steps

        # Step 5: run every tool the LLM asked for and give it the results
        log.info(f"AGENT LOOP step {step}: LLM requested {len(response.tool_calls)} tool call(s)")
        for call in response.tool_calls:
            log.info(f"AGENT running tool {call['name']}({call['args']}) id={call['id']}")
            result = tools_by_name[call["name"]].invoke(call["args"])
            log_block(log, f"AGENT tool result of {call['name']}", str(result))
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            steps.append(f"{call['name']}({call['args']})")

    log.warning(f"AGENT stopped: reached max_steps={max_steps} without a final answer")
    return "Sorry, I could not finish answering. Please contact support@shop.example.com.", steps
