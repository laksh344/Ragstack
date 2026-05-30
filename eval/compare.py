#!/usr/bin/env python
"""A/B comparison of two evaluation runs.

Usage:
  python eval/compare.py eval/results/run_a.json eval/results/run_b.json
  python eval/compare.py eval/results/run_a.json eval/results/run_b.json --by-category
  python eval/compare.py eval/results/run_a.json eval/results/run_b.json --by-difficulty

This is the 'I do A/B testing on my RAG pipeline' interview story:
compare recursive vs semantic chunking, or with/without reranker, etc.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_METRICS = [
    "faithfulness",
    "answer_relevance",
    "retrieval_relevance",
    "citation_accuracy",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGStack A/B eval comparison")
    p.add_argument("run_a", help="Path to first eval result JSON")
    p.add_argument("run_b", help="Path to second eval result JSON")
    p.add_argument("--by-category",   action="store_true", help="Break down by question category")
    p.add_argument("--by-difficulty", action="store_true", help="Break down by difficulty level")
    return p.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    return json.loads(p.read_text())


def _avg(results: list[dict], key: str) -> float:
    values = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    return round(sum(values) / max(len(values), 1), 3)


def _print_table(title: str, rows: list[tuple]) -> None:
    col_w = [max(len(str(r[i])) for r in rows + [(title,)]) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    print(f"\n  {title}")
    print(sep)
    for row in rows:
        line = "|" + "|".join(f" {str(v):<{col_w[i]}} " for i, v in enumerate(row)) + "|"
        print(line)
    print(sep)


def _delta_str(delta: float) -> str:
    if abs(delta) < 0.001:
        return " ±0.000"
    sign = "+" if delta > 0 else ""
    return f" {sign}{delta:+.3f}"


def compare(args: argparse.Namespace) -> None:
    a = _load(args.run_a)
    b = _load(args.run_b)

    label_a = Path(args.run_a).stem
    label_b = Path(args.run_b).stem

    print(f"\n{'='*60}")
    print(f"  RAGStack A/B Comparison")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Overall aggregate comparison
    # ------------------------------------------------------------------
    agg_a = a.get("aggregate", {})
    agg_b = b.get("aggregate", {})

    rows = [("Metric", "Run A", "Run B", "Delta (B-A)", "Winner")]
    for m in _METRICS:
        va = agg_a.get(m, 0.0)
        vb = agg_b.get(m, 0.0)
        delta = round(vb - va, 3)
        winner = "B ✓" if vb > va + 0.001 else ("A ✓" if va > vb + 0.001 else "tie")
        rows.append((m, f"{va:.3f}", f"{vb:.3f}", _delta_str(delta), winner))

    lat_a = agg_a.get("avg_latency_seconds", 0)
    lat_b = agg_b.get("avg_latency_seconds", 0)
    rows.append((
        "avg_latency_s",
        f"{lat_a:.3f}",
        f"{lat_b:.3f}",
        _delta_str(round(lat_b - lat_a, 3)),
        "A ✓" if lat_a < lat_b - 0.01 else ("B ✓" if lat_b < lat_a - 0.01 else "tie"),
    ))

    _print_table("Overall Metrics", rows)

    # ------------------------------------------------------------------
    # Per-category breakdown (optional)
    # ------------------------------------------------------------------
    if args.by_category:
        _breakdown(a["results"], b["results"], label_a, label_b, group_key="category")

    # ------------------------------------------------------------------
    # Per-difficulty breakdown (optional)
    # ------------------------------------------------------------------
    if args.by_difficulty:
        _breakdown(a["results"], b["results"], label_a, label_b, group_key="difficulty")

    print()


def _breakdown(
    results_a: list[dict],
    results_b: list[dict],
    label_a: str,
    label_b: str,
    group_key: str,
) -> None:
    groups = sorted({r.get(group_key) for r in results_a + results_b if r.get(group_key)})
    for group in groups:
        ra = [r for r in results_a if r.get(group_key) == group]
        rb = [r for r in results_b if r.get(group_key) == group]
        rows = [(f"{group_key}={group}", f"A (n={len(ra)})", f"B (n={len(rb)})", "Delta")]
        for m in _METRICS:
            va = _avg(ra, m)
            vb = _avg(rb, m)
            rows.append((m, f"{va:.3f}", f"{vb:.3f}", _delta_str(round(vb - va, 3))))
        _print_table(f"Breakdown by {group_key}", rows)


def main() -> None:
    args = _parse_args()
    compare(args)


if __name__ == "__main__":
    main()
