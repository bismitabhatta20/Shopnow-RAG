"""
Helper: find a good WEAK_RETRIEVAL_DISTANCE for your documents (run once, needs the OpenAI key).

    python -m app.calibrate

It prints the best FAISS distance for questions the documents DO answer and for questions they
do NOT answer. Set WEAK_RETRIEVAL_DISTANCE in .env to a value between the two groups.
"""
from app.debug_logger import end_trace, get_log_file, start_trace
from app.models import get_embedding_model
from app.strategy_selector import best_distance
from app.vector_store import load_vector_store

ANSWERED = ["How long do I have to return headphones?", "How much is shipping?",
            "How do I delete my personal data?", "Can I get a refund for a gift card?"]
NOT_ANSWERED = ["What is the capital of France?", "Do you sell garden furniture?",
                "Can you recommend a good recipe for pasta?"]

if __name__ == "__main__":
    start_trace("CALIBRATE", "python -m app.calibrate")
    embedding_model = get_embedding_model()
    stores = {name: load_vector_store(name, embedding_model) for name in ("knowledge", "legal")}
    for label, questions in (("answered", ANSWERED), ("not answered", NOT_ANSWERED)):
        for question in questions:
            print(f"{label:13} {best_distance(question, stores, 'US'):.3f}  {question}")
    end_trace("calibration finished")
    print(f"Debug log: {get_log_file()}")
