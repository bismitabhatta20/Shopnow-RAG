"""
Project settings.

All values can be changed in the .env file. The only required value is OPENAI_API_KEY.
"""
import os

from dotenv import load_dotenv

# Read the .env file in the project folder (if it exists)
load_dotenv()

# OpenAI models
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")                            # writes the answers
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # turns text into vectors

# Where the documents are, and where the vector indexes are saved
KNOWLEDGE_DIR = "data/knowledge_base"   # policies, FAQ, product guides
LEGAL_DIR = "data/legal_base"           # terms, privacy, consumer rights
INDEX_DIR = "storage"                   # FAISS indexes are saved here

# Chunking and retrieval
CHUNK_SIZE = 800       # max characters per chunk
CHUNK_OVERLAP = 100    # characters shared by neighbouring chunks
TOP_K = 4              # how many chunks to retrieve per search

# Agent
MAX_AGENT_STEPS = 5    # max times the agent may call the model in one question

# LLM judge (checks every answer before the customer sees it)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
JUDGE_MIN_SCORE = 4    # answers scoring below this (1-5) are rewritten or replaced

# Adaptive RAG (query expansion / decomposition only when needed)
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4.1-mini")  # cheaper model that writes search queries
MAX_ATTEMPTS = 3            # max strategies tried for one question
MULTI_PART_MIN_WORDS = 25   # a question this long is treated as multi-part
# FAISS distance: smaller = closer. If even the best chunk is farther than this, the search is "weak".
# Calibrate once with real questions: print the best distances and pick a value between good and bad ones.
WEAK_RETRIEVAL_DISTANCE = float(os.getenv("WEAK_RETRIEVAL_DISTANCE", "1.0"))
