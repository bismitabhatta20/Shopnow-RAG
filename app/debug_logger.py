"""
Debug logger for the whole pipeline.

What it records (one plain-text file per session, in logs/):
    * every function entered and left (arguments, result, time taken)     -> @log_calls
    * every step inside a function (strategy chosen, queries built, ...)  -> log.info(...)
    * every piece of generated / retrieved text in full                    -> log_block(...)
    * every LLM call: the exact messages sent and the text returned        -> LLMDebugCallback
    * every embedding call: what was embedded, vector size, time           -> LoggingEmbeddings
    * a summary of the session when the program exits                      -> atexit

Where to look:
    logs/debug_session_<date>_<time>.log     (written live, so it survives a crash)
    The console shows the same flow at INFO level; the file always has everything (DEBUG).

Settings (optional, in .env):
    DEBUG_LOG_DIR=logs               folder for the log files
    DEBUG_LOG_CONSOLE_LEVEL=INFO     DEBUG = also print full prompts/contexts in the terminal, OFF = file only
    DEBUG_LOG_MAX_CHARS=0            0 = never cut long texts; e.g. 2000 = cut each text block after 2000 chars
    DEBUG_LOG_ENABLED=1              0 = turn the debug logger off completely

How to read a line:
    20:31:02.418 INFO  [Q1] adaptive.answer_adaptively │ │ → ENTER choose_first_strategy(question='...', region='US')
    time         level  trace   module.function       nesting depth  message
    The trace id (Q1, Q2, INGEST ...) groups every line that belongs to one question / one ingestion run.
"""
import atexit
import functools
import logging
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
load_dotenv()
LOG_DIR = os.getenv("DEBUG_LOG_DIR", "logs")
CONSOLE_LEVEL = os.getenv("DEBUG_LOG_CONSOLE_LEVEL", "INFO").upper()
MAX_CHARS = int(os.getenv("DEBUG_LOG_MAX_CHARS", "0"))
ENABLED = os.getenv("DEBUG_LOG_ENABLED", "1") not in ("0", "false", "False", "no")

ROOT_NAME = "shopnow"

# Per-thread state: nesting depth (for the │ indentation) and the current trace id.
# Gradio answers each question in its own worker thread, so thread-local state keeps questions apart.
_state = threading.local()

# Session-wide counters for the summary written at exit
STATS = Counter()
_SESSION = {"start": time.time(), "file": None, "configured": False}
_lock = threading.Lock()


def _depth():
    return getattr(_state, "depth", 0)


def _trace():
    return getattr(_state, "trace", "-")


class _StdoutHandler(logging.StreamHandler):
    """Console handler that always writes to the *current* sys.stdout and never crashes the app
    (pytest / Gradio may swap or close stdout)."""

    def __init__(self):
        super().__init__(sys.stdout)

    def emit(self, record):
        self.stream = sys.stdout
        try:
            super().emit(record)
        except (ValueError, OSError):
            pass

    def flush(self):
        try:
            super().flush()
        except (ValueError, OSError):
            pass


class _ContextFilter(logging.Filter):
    """Adds the trace id and the nesting indentation to every log record."""

    def filter(self, record):
        record.trace = _trace()
        record.indent = "│ " * _depth()
        return True


# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------
def setup_logging():
    """
    Create the session log file and the console output (runs once per program run).

    Returns:
        str or None: Path of the session log file (None when the logger is turned off).
    """
    with _lock:
        if _SESSION["configured"]:
            return _SESSION["file"]
        _SESSION["configured"] = True

        root = logging.getLogger(ROOT_NAME)
        root.propagate = False
        if not ENABLED:
            root.addHandler(logging.NullHandler())
            root.setLevel(logging.CRITICAL + 1)
            return None

        root.setLevel(logging.DEBUG)
        context_filter = _ContextFilter()
        line_format = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-5s [%(trace)s] %(name)s │ %(indent)s%(message)s",
            datefmt="%H:%M:%S")

        # File: everything (DEBUG), written live
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"debug_session_{datetime.now():%Y-%m-%d_%H-%M-%S}.log")
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(line_format)
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)
        _SESSION["file"] = path

        # Console: the flow (INFO by default)
        if CONSOLE_LEVEL != "OFF":
            console = _StdoutHandler()
            console.setLevel(getattr(logging, CONSOLE_LEVEL, logging.INFO))
            console.setFormatter(line_format)
            console.addFilter(context_filter)
            root.addHandler(console)

    _write_session_header(path)
    atexit.register(_write_session_summary)
    return path


def _write_session_header(path):
    """Write the session banner and the active settings (the API key itself is never logged)."""
    log = logging.getLogger(f"{ROOT_NAME}.session")
    log.info("=" * 100)
    log.info(f"DEBUG SESSION STARTED {datetime.now():%Y-%m-%d %H:%M:%S}  ->  log file: {os.path.abspath(path)}")
    log.info(f"python {sys.version.split()[0]} | cwd {os.getcwd()} | pid {os.getpid()}")
    try:
        from app import config
        settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
        for name, value in settings.items():
            log.info(f"config {name} = {value!r}")
    except Exception as error:  # noqa: BLE001 - logging must never break the app
        log.warning(f"could not read app.config: {error}")
    key = os.getenv("OPENAI_API_KEY", "")
    log.info(f"OPENAI_API_KEY is {'set (…' + key[-4:] + ')' if key else 'NOT set'}")
    log.info("=" * 100)


def _write_session_summary():
    """Write a short summary at the end of the session (called automatically at exit)."""
    try:
        _session_summary()
    except Exception:  # noqa: BLE001 - never fail at exit because of logging
        pass


def _session_summary():
    log = logging.getLogger(f"{ROOT_NAME}.session")
    _state.depth = 0
    _state.trace = "END"
    seconds = time.time() - _SESSION["start"]
    log.info("=" * 100)
    log.info(f"DEBUG SESSION ENDED {datetime.now():%Y-%m-%d %H:%M:%S}  (ran {seconds:.1f}s)")
    log.info(f"questions answered     : {STATS['questions']}")
    log.info(f"ingestion runs         : {STATS['ingestions']}")
    log.info(f"functions entered      : {STATS['function_calls']}  (errors: {STATS['function_errors']})")
    log.info(f"LLM calls              : {STATS['llm_calls']}  (errors: {STATS['llm_errors']})")
    log.info(f"LLM tokens in / out    : {STATS['tokens_in']} / {STATS['tokens_out']}")
    log.info(f"embedding calls        : {STATS['embed_calls']}  ({STATS['embedded_texts']} texts)")
    components = {k[4:]: v for k, v in STATS.items() if k.startswith("llm:")}
    if components:
        log.info(f"LLM calls per component: {components}")
    log.info(f"full log saved to      : {os.path.abspath(_SESSION['file'])}")
    log.info("=" * 100)
    for handler in logging.getLogger(ROOT_NAME).handlers:
        handler.flush()


def get_logger(name):
    """
    Get the logger for one module.

    Args:
        name (str): Usually __name__ (e.g. "app.adaptive").

    Returns:
        logging.Logger: Writes to the session file and the console.
    """
    setup_logging()
    return logging.getLogger(f"{ROOT_NAME}.{name.removeprefix('app.')}")


def get_log_file():
    """Path of this session's log file (or None if logging is off)."""
    return setup_logging()


# ---------------------------------------------------------------------------------------------
# Trace ids: group every line of one question / ingestion run
# ---------------------------------------------------------------------------------------------
def start_trace(kind, description=""):
    """
    Start a new trace (e.g. a new customer question) and write a banner for it.

    Args:
        kind (str): "Q" for a question, "INGEST" for an ingestion run, "CALIBRATE" ...
        description (str): Shown in the banner.

    Returns:
        str: The trace id, e.g. "Q3".
    """
    with _lock:
        STATS[f"trace:{kind}"] += 1
        number = STATS[f"trace:{kind}"]
    trace_id = f"{kind}{number}"
    _state.trace = trace_id
    _state.depth = 0
    log = logging.getLogger(f"{ROOT_NAME}.trace")
    log.info("#" * 100)
    log.info(f"### START {trace_id}  {description}")
    log.info("#" * 100)
    return trace_id


def end_trace(result_summary=""):
    """Write the closing banner of the current trace."""
    log = logging.getLogger(f"{ROOT_NAME}.trace")
    _state.depth = 0
    log.info(f"### END {_trace()}  {result_summary}")
    log.info("#" * 100)


# ---------------------------------------------------------------------------------------------
# Helpers to log values and long texts
# ---------------------------------------------------------------------------------------------
def _cut(text, limit):
    if limit and len(text) > limit:
        return text[:limit] + f" …[+{len(text) - limit} chars]"
    return text


def store_label(store):
    """Name of a FAISS store ("knowledge" / "legal") taken from its chunks' metadata."""
    try:
        first = next(iter(store.docstore._dict.values()))
        return f"{first.metadata.get('base', '?')}[{store.index.ntotal} vectors]"
    except Exception:  # noqa: BLE001
        return type(store).__name__


def short(value, limit=300):
    """
    One-line summary of any value, used for function arguments and results.

    Long objects (indexes, models, chunk lists) are summarised; full texts go to log_block().
    """
    try:
        if isinstance(value, str):
            return repr(_cut(value, limit))
        if isinstance(value, Document):
            meta = value.metadata
            return (f"Document(title={meta.get('title')!r}, region={meta.get('region')!r}, "
                    f"section={meta.get('section', '')!r}, {len(value.page_content)} chars)")
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(v, Document) for v in value):
                titles = list(dict.fromkeys(v.metadata.get("title", "?") for v in value))
                return f"[{len(value)} Documents from {titles}]"
            items = ", ".join(short(v, 120) for v in value[:8])
            more = f", …+{len(value) - 8}" if len(value) > 8 else ""
            bracket = "[]" if isinstance(value, list) else "()"
            return f"{bracket[0]}{items}{more}{bracket[1]}"
        if isinstance(value, dict):
            items = ", ".join(f"{k!r}: {short(v, 120)}" for k, v in list(value.items())[:10])
            return "{" + items + "}"
        if hasattr(value, "docstore") and hasattr(value, "index"):
            return f"FAISS({store_label(value)})"
        if hasattr(value, "model_dump") and not hasattr(value, "invoke"):   # Pydantic result (Verdict, QueryList)
            return f"{type(value).__name__}({value.model_dump()})"
        if isinstance(value, (int, float, bool)) or value is None:
            return repr(value)
        if type(value).__module__ == "numpy" and hasattr(value, "item"):   # numpy number (FAISS distance)
            return repr(value.item())
        return f"<{type(value).__name__}>"
    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}>"


def log_block(log, title, text, level=logging.DEBUG):
    """
    Log a multi-line text (prompt, context, answer ...) as a clearly marked block.

    Args:
        log (logging.Logger): The module's logger.
        title (str): What the text is, e.g. "CONTEXT sent to the LLM".
        text (str): The full text. Cut only if DEBUG_LOG_MAX_CHARS is set.
        level (int): logging level (DEBUG by default -> always in the file).
    """
    text = _cut(str(text), MAX_CHARS)
    lines = text.splitlines() or [""]
    log.log(level, f"┌── {title} ({len(text)} chars) " + "─" * max(0, 60 - len(title)))
    for line in lines:
        log.log(level, f"│ {line}")
    log.log(level, "└" + "─" * 70)


def log_documents(log, title, documents, scores=None):
    """
    Log retrieved / created chunks: one summary line each (INFO) and their full text (DEBUG).

    Args:
        log (logging.Logger): The module's logger.
        title (str): e.g. "retrieved chunks".
        documents (list[Document]): The chunks.
        scores (list[float] | None): FAISS distances, if known (smaller = closer).
    """
    log.info(f"{title}: {len(documents)} chunk(s)")
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata
        distance = f" distance={scores[i - 1]:.4f}" if scores else ""
        log.info(f"  [{i}]{distance} title={meta.get('title')!r} region={meta.get('region')!r} "
                 f"base={meta.get('base')!r} section={meta.get('section', '')!r} "
                 f"source={meta.get('source')!r} ({len(doc.page_content)} chars)")
        log_block(log, f"chunk [{i}] text", doc.page_content)


# ---------------------------------------------------------------------------------------------
# Decorator: log every function entered / left
# ---------------------------------------------------------------------------------------------
def log_calls(func):
    """
    Decorator that logs entering a function (with its arguments), leaving it (with its result and
    duration) and any exception (with the full traceback). Nested calls are indented with │.
    """
    log = get_logger(func.__module__)
    name = func.__qualname__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        STATS["function_calls"] += 1
        arguments = [short(a) for a in args] + [f"{k}={short(v)}" for k, v in kwargs.items()]
        log.info(f"→ ENTER {name}({', '.join(arguments)})")
        _state.depth = _depth() + 1
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception:
            _state.depth = max(0, _depth() - 1)
            STATS["function_errors"] += 1
            log.exception(f"✖ ERROR in {name} after {(time.perf_counter() - started) * 1000:.1f} ms")
            raise
        _state.depth = max(0, _depth() - 1)
        log.info(f"← EXIT  {name} ({(time.perf_counter() - started) * 1000:.1f} ms) returned {short(result)}")
        return result

    return wrapper


# ---------------------------------------------------------------------------------------------
# LLM calls: log the exact messages sent and the text generated
# ---------------------------------------------------------------------------------------------
class LLMDebugCallback(BaseCallbackHandler):
    """
    LangChain callback that logs every chat-model call made anywhere in the pipeline:
    which component made it, the model, the full prompt messages, the tools offered,
    the generated text / tool calls / structured output, token usage and time.

    Attach it with debug_config("component name") on a chain or an .invoke() call.
    """

    def __init__(self):
        self.log = get_logger("llm")
        self.runs = {}

    def _component(self, tags, metadata):
        return (metadata or {}).get("component") or ",".join(tags or []) or "unknown"

    def on_chat_model_start(self, serialized, messages, *, run_id, tags=None, metadata=None, **kwargs):
        with _lock:
            STATS["llm_calls"] += 1
            number = STATS["llm_calls"]
        component = self._component(tags, metadata)
        STATS[f"llm:{component}"] += 1
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or params.get("_type") or "?"
        self.runs[run_id] = (number, time.perf_counter(), component)

        self.log.info(f"🤖 LLM CALL #{number} START component={component} model={model} "
                      f"temperature={params.get('temperature', '?')} messages={len(messages[0]) if messages else 0}")
        tools = params.get("tools") or []
        if tools:
            names = [t.get("function", {}).get("name", t.get("name", "?")) for t in tools]
            self.log.info(f"   tools offered to the LLM: {names}")
        if params.get("response_format") or params.get("tool_choice"):
            self.log.info(f"   structured output requested: response_format/tool_choice set")
        for i, message in enumerate(messages[0] if messages else [], start=1):
            extra = ""
            if getattr(message, "tool_calls", None):
                extra = f"\n[tool_calls] {message.tool_calls}"
            if getattr(message, "tool_call_id", None):
                extra = f"\n[tool_call_id] {message.tool_call_id}"
            log_block(self.log, f"LLM #{number} INPUT message {i} [{message.type}]",
                      f"{message.content}{extra}")

    def on_llm_start(self, serialized, prompts, *, run_id, tags=None, metadata=None, **kwargs):
        """Non-chat (plain text) models - not used by this project, logged for completeness."""
        with _lock:
            STATS["llm_calls"] += 1
            number = STATS["llm_calls"]
        component = self._component(tags, metadata)
        self.runs[run_id] = (number, time.perf_counter(), component)
        self.log.info(f"🤖 LLM CALL #{number} START component={component} (text model)")
        for i, prompt in enumerate(prompts, start=1):
            log_block(self.log, f"LLM #{number} INPUT prompt {i}", prompt)

    def on_llm_end(self, response, *, run_id, **kwargs):
        number, started, component = self.runs.pop(run_id, ("?", time.perf_counter(), "?"))
        seconds = time.perf_counter() - started
        usage_in = usage_out = 0
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                text = generation.text or (getattr(message, "content", "") if message else "")
                if text:
                    log_block(self.log, f"LLM #{number} OUTPUT text", text)
                if message is not None:
                    if getattr(message, "tool_calls", None):
                        for call in message.tool_calls:
                            self.log.info(f"   LLM #{number} requested tool {call.get('name')}({call.get('args')})")
                    parsed = (getattr(message, "additional_kwargs", {}) or {}).get("parsed")
                    if parsed is not None:
                        log_block(self.log, f"LLM #{number} OUTPUT structured", short(parsed, 0))
                    usage = getattr(message, "usage_metadata", None) or {}
                    usage_in += usage.get("input_tokens", 0)
                    usage_out += usage.get("output_tokens", 0)
        if not (usage_in or usage_out):
            token_usage = (response.llm_output or {}).get("token_usage") or {}
            usage_in = token_usage.get("prompt_tokens", 0)
            usage_out = token_usage.get("completion_tokens", 0)
        STATS["tokens_in"] += usage_in
        STATS["tokens_out"] += usage_out
        self.log.info(f"🤖 LLM CALL #{number} END component={component} in {seconds:.2f}s "
                      f"tokens in/out={usage_in}/{usage_out}")

    def on_llm_error(self, error, *, run_id, **kwargs):
        number, started, component = self.runs.pop(run_id, ("?", time.perf_counter(), "?"))
        STATS["llm_errors"] += 1
        self.log.error(f"🤖 LLM CALL #{number} FAILED component={component} after "
                       f"{time.perf_counter() - started:.2f}s: {type(error).__name__}: {error}")


LLM_CALLBACK = LLMDebugCallback()


def debug_config(component, **metadata):
    """
    LangChain config that turns on LLM logging and labels the calls with a component name.

    Use it on a chain:   chain.with_config(debug_config("qa_chain"))
    or on one call:      llm.invoke(messages, config=debug_config("agent", step=2))
    """
    return {"callbacks": [LLM_CALLBACK], "metadata": {"component": component, **metadata},
            "run_name": component}


# ---------------------------------------------------------------------------------------------
# Embeddings: log what is embedded
# ---------------------------------------------------------------------------------------------
class LoggingEmbeddings(Embeddings):
    """Wraps any embedding model and logs every call (texts, number of vectors, size, time)."""

    def __init__(self, inner):
        self.inner = inner
        self.log = get_logger("embeddings")
        self.model_name = getattr(inner, "model", type(inner).__name__)

    def embed_documents(self, texts):
        STATS["embed_calls"] += 1
        STATS["embedded_texts"] += len(texts)
        self.log.info(f"🔢 EMBED {len(texts)} document chunk(s) with {self.model_name} …")
        started = time.perf_counter()
        vectors = self.inner.embed_documents(texts)
        size = len(vectors[0]) if vectors else 0
        self.log.info(f"🔢 EMBED done: {len(vectors)} vectors × {size} dims in {time.perf_counter() - started:.2f}s")
        return vectors

    def embed_query(self, text):
        STATS["embed_calls"] += 1
        STATS["embedded_texts"] += 1
        started = time.perf_counter()
        vector = self.inner.embed_query(text)
        preview = ", ".join(f"{v:.4f}" for v in vector[:5])
        self.log.info(f"🔢 EMBED query {short(text, 200)} -> {len(vector)} dims [{preview}, …] "
                      f"in {(time.perf_counter() - started) * 1000:.0f} ms")
        return vector

    def __getattr__(self, name):
        if name == "inner":            # avoid endless recursion while copying / unpickling
            raise AttributeError(name)
        return getattr(self.inner, name)


def wrap_embeddings(embedding_model):
    """Wrap an embedding model with LoggingEmbeddings (only once)."""
    if embedding_model is None or isinstance(embedding_model, LoggingEmbeddings):
        return embedding_model
    return LoggingEmbeddings(embedding_model)
