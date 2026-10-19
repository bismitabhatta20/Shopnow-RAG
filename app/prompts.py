"""
Step 8 - Prompts: the instructions we send to the LLM.
"""
from langchain_core.prompts import ChatPromptTemplate

from app.debug_logger import get_logger, log_block, log_calls

log = get_logger(__name__)


def _logged(name, prompt):
    """Log a newly created prompt template: its variables and the text of every message."""
    log.info(f"prompt {name!r} created: variables={sorted(prompt.input_variables)}, {len(prompt.messages)} message(s)")
    for message in prompt.messages:
        role = type(message).__name__.replace("MessagePromptTemplate", "").lower()
        log_block(log, f"prompt {name!r} template [{role}]", message.prompt.template)
    return prompt


@log_calls
def create_qa_prompt():
    """
    Create the prompt for the simple RAG chain: rules + context + question.

    Returns:
        ChatPromptTemplate: Needs the variables region, context, question and feedback.
    """
    system = """You are ShopNow's customer support assistant. The customer is in region {region}.
Rules:
1. Answer ONLY from the context. If the answer is not there, say you don't know and suggest
   contacting support@shop.example.com.
2. Give exact numbers, time limits and conditions. Cite sources like [1] or [2].
3. If EU or UK law gives the customer more rights than the store policy, say so.
4. You cannot view or change orders. Explain how the customer can do it in their account.
5. If you used a legal document, end with: "This is general information, not legal advice."
"""
    return _logged('qa', ChatPromptTemplate.from_messages([
        ("system", system),
        # {feedback} is empty normally; when the judge rejects an answer it holds the judge's comments
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\n{feedback}"),
    ]))


@log_calls
def create_agent_prompt():
    """
    Create the prompt for the agent. The agent finds its own context by calling search tools.

    Returns:
        ChatPromptTemplate: Needs the variables region and question.
    """
    system = """You are ShopNow's customer support agent. The customer is in region {region}.
How to work:
1. Search before you answer. Store rules and product help -> search_knowledge_base.
   Rights, terms, privacy -> search_legal_base. You may call both.
2. If the results do not answer the question, search again with other words.
3. Use calculate_date for any date or deadline.
4. Use escalate_to_human only if the customer asks for a person or has a dispute.
Rules:
1. Answer ONLY from the search results. If the answer is not there, say you don't know and suggest
   contacting support@shop.example.com.
2. Give exact numbers, time limits and conditions. Cite sources like [1] or [2].
3. If EU or UK law gives the customer more rights than the store policy, say so.
4. You cannot view or change orders. Explain how the customer can do it in their account.
5. If you used a legal document, end with: "This is general information, not legal advice."
"""
    return _logged('agent', ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ]))


@log_calls
def create_judge_prompt():
    """
    Create the prompt for the LLM judge. The judge reviews an answer; it never writes one.

    Returns:
        ChatPromptTemplate: Needs the variables question, context and answer.
    """
    system = """You are a strict reviewer of customer-support answers.
You get a customer question, the context (document passages) and the answer that was written from it.
Check three things:
1. context_relevant: does the context contain the information needed to answer the question?
2. answer_relevant: does the answer directly address what the customer asked?
3. grounded: is every fact in the answer (numbers, time limits, conditions) supported by the context?
   An answer that honestly says "I don't know" is grounded. Ignore citation numbers like [1].
Then give a score from 1 (unusable) to 5 (correct, complete and supported),
and a one-sentence reason that says what is wrong, if anything."""
    return _logged('judge', ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer to review:\n{answer}"),
    ]))


@log_calls
def create_expansion_prompt():
    """
    Create the prompt for QUERY EXPANSION: rewrite one question in the words the documents use.

    Returns:
        ChatPromptTemplate: Needs the variable question.
    """
    system = """You write search queries for a store's help documents (policies, FAQ, product guides,
terms, privacy and consumer-rights pages). Rewrite the customer's question as 3 short search queries
that use the words such documents use (e.g. "send it back" -> "return window").
Keep the meaning. Do not answer the question."""
    return _logged('expansion', ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ]))


@log_calls
def create_decomposition_prompt():
    """
    Create the prompt for QUERY DECOMPOSITION: split a multi-part question into simple sub-questions.

    Returns:
        ChatPromptTemplate: Needs the variable question.
    """
    system = """Split the customer's question into simple, self-contained sub-questions, one per topic
(2 to 4). Each sub-question must make sense on its own. If the question has only one topic,
return it unchanged as the only item. Do not answer the question."""
    return _logged('decomposition', ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ]))
