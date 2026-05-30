"""Evaluation API — run the eval suite and compare configurations.

POST /api/v1/evaluate          — run full eval suite against golden_qa
GET  /api/v1/evaluate/results  — list stored eval run summaries
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = structlog.get_logger()
router = APIRouter()

_RESULTS_DIR = Path(__file__).parent.parent.parent / "eval" / "results"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    dataset: str = "golden_qa"
    config: dict = {}
    compare_to: str | None = None   # run_id of a previous result to compare


class EvaluateResponse(BaseModel):
    run_id: str
    status: str   # "running" | "complete" | "error"
    aggregate: dict = {}
    comparison: dict | None = None
    results_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """Trigger the evaluation suite.

    Runs synchronously (small dataset) and returns results immediately.
    For large datasets, the heavy lifting could be moved to a background task.
    """
    from backend.observability.datasets import load_golden_qa, run_evaluation  # deferred

    if request.dataset != "golden_qa":
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}")

    try:
        examples = load_golden_qa()
        eval_result = await run_evaluation(examples=examples, config=request.config)
    except Exception as exc:
        logger.error("evaluate.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    run_id = eval_result["run_id"]
    _persist_result(run_id, eval_result)

    comparison: dict | None = None
    if request.compare_to:
        try:
            comparison = _compare_results(eval_result, request.compare_to)
        except Exception as exc:
            logger.warning("evaluate.compare_error", error=str(exc))

    logger.info("evaluate.complete", run_id=run_id, aggregate=eval_result["aggregate"])
    return EvaluateResponse(
        run_id=run_id,
        status="complete",
        aggregate=eval_result["aggregate"],
        comparison=comparison,
        results_path=str(_result_path(run_id)),
    )


@router.get("/evaluate/results")
async def list_results():
    """List all stored evaluation run summaries."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for path in sorted(_RESULTS_DIR.glob("*.json"), reverse=True)[:20]:
        try:
            data = json.loads(path.read_text())
            summaries.append({
                "run_id":    data.get("run_id"),
                "timestamp": data.get("timestamp"),
                "aggregate": data.get("aggregate", {}),
                "file":      path.name,
            })
        except Exception:
            pass
    return {"results": summaries, "count": len(summaries)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_path(run_id: str) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _RESULTS_DIR / f"eval_{ts}_{run_id[:8]}.json"


def _persist_result(run_id: str, result: dict) -> None:
    path = _result_path(run_id)
    payload = {
        **result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("evaluate.saved", path=str(path))


def _compare_results(current: dict, compare_to_run_id: str) -> dict:
    """Load a previous run and build a side-by-side diff."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = list(_RESULTS_DIR.glob(f"eval_*_{compare_to_run_id[:8]}.json"))
    if not matches:
        raise FileNotFoundError(f"No result file found for run_id prefix {compare_to_run_id[:8]}")

    prev = json.loads(matches[0].read_text())
    curr_agg = current.get("aggregate", {})
    prev_agg = prev.get("aggregate", {})

    metrics = ["faithfulness", "answer_relevance", "retrieval_relevance", "citation_accuracy"]
    diff: dict = {}
    for m in metrics:
        curr_val = curr_agg.get(m, 0)
        prev_val = prev_agg.get(m, 0)
        diff[m] = {
            "current":  curr_val,
            "previous": prev_val,
            "delta":    round(curr_val - prev_val, 3),
            "improved": curr_val > prev_val,
        }
    return {"baseline_run_id": compare_to_run_id, "metrics": diff}
