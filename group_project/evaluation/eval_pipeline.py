"""Run a small RAGAS evaluation and save an auditable Markdown report.

Default is five cases to avoid exhausting a classroom API quota. Use
``python -m group_project.evaluation.eval_pipeline --limit 15`` for all cases.
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
load_dotenv(ROOT / ".env")


def load_golden_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _ragas_clients():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Configure OPENROUTER_API_KEY or OPENAI_API_KEY in .env")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini") if base_url else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, api_key=key, base_url=base_url, temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key, base_url=base_url)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def evaluate_with_ragas(cases):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from src.task10_generation import generate_with_citation

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in cases:
        result = generate_with_citation(item["question"], top_k=5)
        rows["question"].append(item["question"])
        rows["answer"].append(result["answer"])
        rows["contexts"].append([source["content"] for source in result.get("sources", [])])
        rows["ground_truth"].append(item["expected_answer"])
    llm, embeddings = _ragas_clients()
    result = evaluate(Dataset.from_dict(rows), metrics=[faithfulness, answer_relevancy, context_recall, context_precision], llm=llm, embeddings=embeddings, raise_exceptions=False)
    return result, rows


def main(limit: int):
    cases = load_golden_dataset()[:limit]
    print(f"Loaded {len(cases)} test cases")
    result, _ = evaluate_with_ragas(cases)
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    RESULTS_PATH.write_text(
        "# RAG Evaluation Results\n\n"
        f"Framework: RAGAS 0.1.21 | evaluated cases: {len(cases)}\n\n"
        "| Metric | Score |\n|---|---:|\n"
        f"| Faithfulness | {scores.get('faithfulness', 0):.4f} |\n"
        f"| Answer relevancy | {scores.get('answer_relevancy', 0):.4f} |\n"
        f"| Context recall | {scores.get('context_recall', 0):.4f} |\n"
        f"| Context precision | {scores.get('context_precision', 0):.4f} |\n\n"
        "## Notes\n\nConfig A is the current hybrid retrieval pipeline (dense + BM25 + RRF). "
        "Run a separate dense-only pass before claiming an A/B comparison.\n",
        encoding="utf-8",
    )
    print("Saved RAGAS metrics to", RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    main(parser.parse_args().limit)
