"""LangSmith evaluation dataset management.

Loads the local golden_qa.json, optionally pushes it to LangSmith,
and runs the evaluation suite against the live agent pipeline.
"""

import json
import time
from pathlib import Path
from uuid import uuid4

import structlog

logger = structlog.get_logger()

_GOLDEN_QA_PATH = Path(__file__).parent.parent.parent / "eval" / "datasets" / "golden_qa.json"
_LANGSMITH_DATASET_NAME = "ragstack-golden-qa"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_qa() -> list[dict]:
    """Load the local golden QA dataset from disk."""
    if not _GOLDEN_QA_PATH.exists():
        raise FileNotFoundError(f"Golden QA dataset not found: {_GOLDEN_QA_PATH}")
    with _GOLDEN_QA_PATH.open() as f:
        data = json.load(f)
    logger.info("datasets.loaded", path=str(_GOLDEN_QA_PATH), count=len(data))
    return data


# ---------------------------------------------------------------------------
# LangSmith dataset management
# ---------------------------------------------------------------------------

def push_to_langsmith(examples: list[dict] | None = None) -> str | None:
    """Create or update the LangSmith evaluation dataset.

    Returns the dataset ID if successful, None if LangSmith is unavailable.
    """
    from backend.config import settings  # deferred

    if not settings.langchain_api_key:
        logger.warning("datasets.langsmith_no_key")
        return None

    try:
        from langsmith import Client  # noqa: PLC0415

        client = Client()
        items = examples or load_golden_qa()

        # Create dataset (idempotent — LangSmith returns existing if name matches)
        dataset = client.create_dataset(
            dataset_name=_LANGSMITH_DATASET_NAME,
            description="Golden QA pairs for RAGStack evaluation",
        )

        # Add examples
        for item in items:
            client.create_example(
                inputs={"question": item["question"]},
                outputs={
                    "answer": item["expected_answer"],
                    "source_doc": item.get("source_doc", ""),
                    "difficulty": item.get("difficulty", ""),
                    "category": item.get("category", ""),
                },
                dataset_id=dataset.id,
            )

        logger.info("datasets.langsmith_pushed", dataset_id=str(dataset.id), count=len(items))
        return str(dataset.id)

    except Exception as exc:
        logger.error("datasets.langsmith_error", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Programmatic evaluation runner
# ---------------------------------------------------------------------------

async def run_evaluation(
    examples: list[dict] | None = None,
    config: dict | None = None,
) -> dict:
    """Run the full eval suite and return structured results.

    Args:
        examples: QA pairs to evaluate. Defaults to golden_qa.json.
        config:   Optional overrides (e.g. {"top_k": 5}).

    Returns:
        Dict with "results" (per-question), "aggregate" (mean scores),
        and "run_id".
    """
    from backend.agent.graph import graph          # deferred
    from backend.observability.evaluators import (  # deferred
        evaluate_faithfulness,
        evaluate_answer_relevance,
        evaluate_retrieval_relevance,
        evaluate_citation_accuracy,
    )

    cfg = config or {}
    items = examples or load_golden_qa()
    run_id = str(uuid4())
    results: list[dict] = []

    for item in items:
        question = item["question"]
        expected = item.get("expected_answer", "")
        t0 = time.perf_counter()

        try:
            state = await graph.ainvoke({
                "query": question,
                "messages": [],
                "route": "",
                "iteration_count": 0,
                "retrieved_docs": [],
                "needs_web_search": False,
                "search_results": [],
                "response": "",
                "citations": [],
                "guardrail_flags": [],
            })

            response  = state.get("response", "")
            citations = state.get("citations", [])
            chunks    = state.get("retrieved_docs", [])
            context   = " ".join(c.get("content", "") for c in chunks)
            latency   = round(time.perf_counter() - t0, 3)

            faith_score   = await evaluate_faithfulness(question, response, context, use_llm=False)
            rel_score     = await evaluate_answer_relevance(question, response, use_llm=False)
            retr_score    = evaluate_retrieval_relevance(question, chunks)
            cite_score    = evaluate_citation_accuracy(citations, chunks)

            results.append({
                "id":                  item.get("id"),
                "question":            question,
                "expected_answer":     expected,
                "actual_response":     response,
                "latency_seconds":     latency,
                "faithfulness":        faith_score.score,
                "answer_relevance":    rel_score.score,
                "retrieval_relevance": retr_score.score,
                "citation_accuracy":   cite_score.score,
                "guardrail_flags":     state.get("guardrail_flags", []),
                "difficulty":          item.get("difficulty"),
                "category":            item.get("category"),
                "error":               None,
            })

        except Exception as exc:
            logger.error("datasets.eval_item_error", question=question[:60], error=str(exc))
            results.append({
                "id": item.get("id"),
                "question": question,
                "error": str(exc),
            })

    aggregate = _aggregate(results)
    logger.info("datasets.eval_complete", run_id=run_id, items=len(results), **aggregate)
    return {"run_id": run_id, "results": results, "aggregate": aggregate}


def _aggregate(results: list[dict]) -> dict:
    """Compute mean scores across all successful eval items."""
    keys = ["faithfulness", "answer_relevance", "retrieval_relevance", "citation_accuracy"]
    agg: dict = {}
    for key in keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        agg[key] = round(sum(values) / max(len(values), 1), 3)
    agg["total_items"] = len(results)
    agg["error_count"]  = sum(1 for r in results if r.get("error"))
    agg["avg_latency_seconds"] = round(
        sum(r.get("latency_seconds", 0) for r in results if not r.get("error"))
        / max(sum(1 for r in results if not r.get("error")), 1),
        3,
    )
    return agg
