"""
Step 3 - Pre-processing.

    Documents : remove extra blank lines and spaces before chunking.
    Questions : remove card numbers so they are never sent to OpenAI.
"""
import re

from app.debug_logger import get_logger, log_calls

log = get_logger(__name__)

CARD_PATTERN = r"\b(?:\d[ -]?){13,19}\b"


@log_calls
def clean_text(text):
    """
    Tidy up document text.

    Args:
        text (str): Raw document text.

    Returns:
        str: Text without trailing spaces and without runs of blank lines.
    """
    before = len(text)
    text = re.sub(r"[ \t]+\n", "\n", text)    # spaces at the end of lines
    text = re.sub(r"\n{3,}", "\n\n", text)    # 3+ newlines -> one blank line
    text = text.strip()
    log.info(f"cleaned text: {before} -> {len(text)} chars")
    return text


def remove_card_numbers(question):
    """
    Replace anything that looks like a payment card number (13-19 digits) with a placeholder.

    Args:
        question (str): The customer's question.

    Returns:
        str: The question with card numbers removed.
    """
    # Not decorated with @log_calls on purpose: that would write the raw question - card number
    # included - to the log file. Only the cleaned question is logged.
    log.info(f"→ ENTER remove_card_numbers(<raw question, {len(question)} chars, not logged>)")
    cleaned, count = re.subn(CARD_PATTERN, "[card number removed]", question)
    log.info(f"← EXIT  remove_card_numbers: {count} card number(s) removed -> {cleaned!r}")
    return cleaned
