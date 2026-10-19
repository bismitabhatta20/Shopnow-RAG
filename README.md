# ShopNow Support — a simple RAG assistant (LangChain + OpenAI + Gradio)

Answers customer questions from two sets of documents:
- **Knowledge base**: store policies, FAQ, product guides (`data/knowledge_base/`)
- **Legal base**: terms, privacy, EU/UK/US consumer rights (`data/legal_base/`)

## How it works

```
INGESTION (once)   load_documents -> clean_text -> chunk_documents -> create_vector_store (embed) -> save
                   app/loader.py     preprocess.py   chunker.py         vector_store.py             (storage/)

QUESTION           question -> retrieve chunks -> context + question -> OpenAI LLM -> answer
  Simple chain     rag_chain.generate_answer(): always searches both bases, then calls the LLM once
  Agent            agent.run_agent(): the LLM calls search tools itself, in a loop, then answers
  LLM judge        judge.verify_answer(): checks context relevance, answer relevance and groundedness;
                   rewrites once with feedback, or sends a safe message - before the customer sees anything
  Adaptive RAG     adaptive.answer_adaptively(): picks a query strategy with free checks
                   (direct / expansion / decomposition / decompose+expand), answers, lets the judge check,
                   and switches strategy only if the answer is rejected (max 3 tries)
```

## Files (read them in this order)

| Step | File | What it does |
|---|---|---|
| 0 | `app/config.py` | Settings (models, folders, chunk size) |
| 1 | `app/models.py` | `get_llm()`, `get_embedding_model()` — the two OpenAI models |
| 2 | `app/loader.py` | `load_documents()` — reads the Markdown files |
| 3 | `app/preprocess.py` | `clean_text()`, `remove_card_numbers()` |
| 4 | `app/chunker.py` | `chunk_documents()` — split by heading, then by size |
| 5-6 | `app/vector_store.py` | `create_vector_store()` (embed + index with FAISS), save / load |
| — | `app/ingest.py` | `build_all_indexes()` — runs steps 2 to 6 |
| 7 | `app/retriever.py` | `retrieve()` — similar chunks for the region; `format_context()` |
| 8 | `app/prompts.py` | `create_qa_prompt()`, `create_agent_prompt()` |
| 9 | `app/rag_chain.py` | `create_qa_chain()`, `generate_answer()` — simple RAG |
| 10 | `app/agent.py` | `create_tools()`, `run_agent()` — tool calling in a loop |
| 11 | `app/judge.py` | `Verdict`, `create_judge()`, `verify_answer()` — LLM as a judge |
| 12 | `app/ui.py` | Gradio app: `answer_question()` (answer → judge → customer) |
| 13a | `app/query_strategies.py` | `build_queries()` for the 4 strategies, `retrieve_for_queries()` |
| 13b | `app/strategy_selector.py` | Free checks `is_multi_part()`, `best_distance()`; `choose_first_strategy()`, `next_strategy()` |
| 13c | `app/adaptive.py` | `answer_adaptively()` — the adaptive loop |
| — | `app/debug_logger.py` | Debug logger: every function, step, prompt, LLM output and judge verdict → `logs/` |
| — | `app/calibrate.py` | `python -m app.calibrate` — helps you set WEAK_RETRIEVAL_DISTANCE |
| — | `tests/` | 27 offline tests: every step, every judge path, every strategy and escalation (fake models) |

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # put your OPENAI_API_KEY in .env
python -m app.ingest            # build the indexes (or use the "Build index" button)
python -m app.ui                # open http://localhost:7860
pytest                          # offline tests, no API key needed
```

## Debug log

Every run writes one log file: `logs/debug_session_<date>_<time>.log` (its path is printed at start-up).
It contains, in order and grouped per question (`[Q1]`, `[Q2]`, …) or ingestion run (`[INGEST1]`):

- every function entered/left (`→ ENTER` / `← EXIT`) with its arguments, result and time; nesting shown with `│`
- every pipeline step (e.g. `ADAPTIVE ATTEMPT 1/3`, `FREE CHECK 1/2`, `🎯 STRATEGY SELECTED`)
- the queries each strategy produced, every retrieved chunk with its FAISS distance, and the full context
- every LLM call (`🤖 LLM CALL #n`): component, model, each input message in full, the generated output, tokens, time
- every judge verdict (`⚖ JUDGE VERDICT`) and the decision taken from it
- a session summary at exit (questions, LLM calls per component, tokens, embedding calls, errors)

The terminal shows the same flow at INFO level; full texts (prompts, contexts, chunk texts) are in the file.
Set `DEBUG_LOG_CONSOLE_LEVEL=DEBUG` in `.env` to print them in the terminal too. Card numbers are never logged:
only the cleaned question is written.
