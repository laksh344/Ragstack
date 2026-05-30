#!/usr/bin/env python
"""CLI evaluation runner — executes the full eval suite and saves results.

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --dataset golden_qa --output eval/results/
  python eval/run_eval.py --subset 5           # run only first 5 items
  python eval/run_eval.py --no-llm             # skip LLM-judge evaluators
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Ensure the project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGStack evaluation runner")
    p.add_argument("--dataset",  default="golden_qa", help="Dataset name (default: golden_qa)")
    p.add_argument("--output",   default="eval/results", help="Output directory for result JSON")
    p.add_argument("--subset",   type=int, default=None, help="Evaluate only the first N items")
    p.add_argument("--no-llm",   action="store_true",   help="Skip LLM-judge evaluators (faster)")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    from backend.observability.datasets import load_golden_qa, _aggregate
    from backend.agent.graph import graph
    from backend.observability.evaluators import (
        evaluate_faithfulness,
        evaluate_answer_relevance,
        evaluate_retrieval_relevance,
        evaluate_citation_accuracy,
    )

    examples = load_golden_qa()
    if args.subset:
        examples = examples[:args.subset]

    use_llm = not args.no_llm
    print(f"\n{'='*60}")
    print(f"  RAGStack Evaluation Runner")
    print(f"  Dataset : {args.dataset}  ({len(examples)} items)")
    print(f"  LLM judge: {'enabled' if use_llm else 'disabled (overlap fallback)'}")
    print(f"{'='*60}\n")

    import time
    run_id  = str(uuid4())
    results: list[dict] = []

    for i, item in enumerate(examples, 1):
        question = item["question"]
        print(f"[{i:02d}/{len(examples)}] {question[:70]}...")
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

            faith = await evaluate_faithfulness(question, response, context, use_llm=use_llm)
            rel   = await evaluate_answer_relevance(question, response, use_llm=use_llm)
            retr  = evaluate_retrieval_relevance(question, chunks)
            cite  = evaluate_citation_accuracy(citations, chunks)

            row = {
                "id":                  item.get("id"),
                "question":            question,
                "expected_answer":     item.get("expected_answer", ""),
                "actual_response":     response,
                "latency_seconds":     latency,
                "faithfulness":        faith.score,
                "answer_relevance":    rel.score,
                "retrieval_relevance": retr.score,
                "citation_accuracy":   cite.score,
                "guardrail_flags":     state.get("guardrail_flags", []),
                "difficulty":          item.get("difficulty"),
                "category":            item.get("category"),
                "error":               None,
            }
            results.append(row)

            print(
                f"       faith={faith.score:.2f}  rel={rel.score:.2f}  "
                f"retr={retr.score:.2f}  cite={cite.score:.2f}  "
                f"latency={latency}s"
            )

        except Exception as exc:
            latency = round(time.perf_counter() - t0, 3)
            print(f"       ERROR: {exc}")
            results.append({"id": item.get("id"), "question": question, "error": str(exc)})

    # ------------------------------------------------------------------
    # Aggregate + save
    # ------------------------------------------------------------------
    aggregate = _aggregate(results)
    output = {
        "run_id":    run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset": args.dataset,
            "subset":  args.subset,
            "use_llm": use_llm,
        },
        "aggregate": aggregate,
        "results":   results,
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}_{run_id[:8]}.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    for k, v in aggregate.items():
        print(f"  {k:<30} {v}")
    print(f"\n  Saved → {out_path}")
    print(f"  Run ID: {run_id}\n")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
