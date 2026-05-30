"""Unit tests for the retrieval layer.

All tests are pure Python — no Qdrant, Elasticsearch, or Cohere required.
Integration tests (actual network calls) belong in a separate suite run
only when Docker services are available.
"""


from backend.retrieval import SearchResult
from backend.retrieval.hybrid import _RRF_K, reciprocal_rank_fusion
from backend.retrieval.keyword_store import _MAX_BM25_SCORE, _hit_to_result
from backend.retrieval.reranker import _mark_reranked
from backend.retrieval.vector_store import _build_filter, _point_to_result

# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult(chunk_id="c1", content="hello", source_file="a.txt")
        assert r.score == 0.0
        assert r.source == "vector"
        assert r.metadata == {}

    def test_model_copy_is_independent(self):
        r = SearchResult(chunk_id="c1", content="hello", source_file="a.txt", score=0.9)
        c = r.model_copy()
        c.score = 0.1
        assert r.score == 0.9  # original unchanged


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _make_results(ids: list[str], start_score: float = 1.0) -> list[SearchResult]:
    return [
        SearchResult(chunk_id=cid, content=cid, source_file="f.txt", score=start_score - i * 0.1)
        for i, cid in enumerate(ids)
    ]


class TestRRF:
    def test_single_list_ranking_preserved(self):
        results = _make_results(["a", "b", "c"])
        fused = reciprocal_rank_fusion([results])
        ids = [r.chunk_id for r in fused]
        assert ids == ["a", "b", "c"]

    def test_two_lists_boost_overlap(self):
        # "b" appears in both lists → should rank above "a" (rank-1 in list1 only)
        list1 = _make_results(["a", "b", "c"])
        list2 = _make_results(["b", "d", "e"])
        fused = reciprocal_rank_fusion([list1, list2])
        # "b" is rank-2 in list1 + rank-1 in list2; "a" is rank-1 in list1 only
        b_score = 1 / (_RRF_K + 2) + 1 / (_RRF_K + 1)
        a_score = 1 / (_RRF_K + 1)
        assert b_score > a_score
        assert fused[0].chunk_id == "b"

    def test_scores_are_rrf_formula(self):
        results = _make_results(["x", "y"])
        fused = reciprocal_rank_fusion([results])
        # Scores are rounded to 6 decimal places; allow 1e-5 tolerance.
        assert abs(fused[0].score - 1 / (_RRF_K + 1)) < 1e-5
        assert abs(fused[1].score - 1 / (_RRF_K + 2)) < 1e-5

    def test_source_set_to_hybrid(self):
        fused = reciprocal_rank_fusion([_make_results(["a"])])
        assert fused[0].source == "hybrid"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_disjoint_lists_all_included(self):
        list1 = _make_results(["a", "b"])
        list2 = _make_results(["c", "d"])
        fused = reciprocal_rank_fusion([list1, list2])
        ids = {r.chunk_id for r in fused}
        assert ids == {"a", "b", "c", "d"}

    def test_custom_k(self):
        results = _make_results(["x"])
        fused_default = reciprocal_rank_fusion([results])
        fused_custom = reciprocal_rank_fusion([results], k=10)
        # Smaller k → higher score for same rank
        assert fused_custom[0].score > fused_default[0].score

    def test_does_not_mutate_input(self):
        original = _make_results(["a", "b"])
        original_scores = [r.score for r in original]
        reciprocal_rank_fusion([original])
        assert [r.score for r in original] == original_scores


# ---------------------------------------------------------------------------
# Vector store helpers (no network)
# ---------------------------------------------------------------------------


class TestVectorStoreHelpers:
    def test_build_filter_single(self):
        from qdrant_client.models import FieldCondition, Filter

        f = _build_filter({"source_file": "report.pdf"})
        assert isinstance(f, Filter)
        assert len(f.must) == 1
        assert isinstance(f.must[0], FieldCondition)
        assert f.must[0].key == "source_file"

    def test_build_filter_multiple(self):
        f = _build_filter({"source_file": "a.pdf", "file_type": "pdf"})
        assert len(f.must) == 2

    def test_point_to_result_maps_payload(self):
        class FakePoint:
            id = "abc"
            score = 0.87
            payload = {
                "chunk_id": "c42",
                "content": "hello world",
                "source_file": "doc.pdf",
                "page_number": 3,
                "chunk_index": 1,
                "title": "Doc",
                "file_type": "pdf",
                "char_count": 11,
                "token_estimate": 3,
                "chunking_strategy": "recursive",
            }

        r = _point_to_result(FakePoint())
        assert r.chunk_id == "c42"
        assert r.content == "hello world"
        assert r.score == 0.87
        assert r.source == "vector"
        assert r.page_number == 3

    def test_point_to_result_missing_payload(self):
        class FakePoint:
            id = "xyz"
            score = 0.5
            payload = None

        r = _point_to_result(FakePoint())
        assert r.chunk_id == "xyz"  # falls back to point id
        assert r.content == ""


# ---------------------------------------------------------------------------
# Keyword store helpers (no network)
# ---------------------------------------------------------------------------


class TestKeywordStoreHelpers:
    def _make_hit(self, score: float, source: dict | None = None) -> dict:
        return {
            "_id": "es-id",
            "_score": score,
            "_source": source or {
                "chunk_id": "ck1",
                "content": "some text",
                "source_file": "a.txt",
                "page_number": 1,
                "chunk_index": 0,
                "title": "A",
                "file_type": "txt",
                "metadata": {},
            },
        }

    def test_score_normalised(self):
        r = _hit_to_result(self._make_hit(_MAX_BM25_SCORE))
        assert r.score == 1.0

    def test_score_capped_at_one(self):
        r = _hit_to_result(self._make_hit(_MAX_BM25_SCORE * 2))
        assert r.score == 1.0

    def test_source_is_keyword(self):
        r = _hit_to_result(self._make_hit(5.0))
        assert r.source == "keyword"

    def test_missing_chunk_id_falls_back_to_es_id(self):
        r = _hit_to_result(self._make_hit(1.0, source={"content": "x"}))
        assert r.chunk_id == "es-id"


# ---------------------------------------------------------------------------
# Reranker helpers (no network)
# ---------------------------------------------------------------------------


class TestRerankerHelpers:
    def test_mark_reranked_sets_source(self):
        results = _make_results(["a", "b"])
        tagged = _mark_reranked(results)
        assert all(r.source == "reranked" for r in tagged)

    def test_mark_reranked_preserves_scores(self):
        results = _make_results(["a"])
        original_score = results[0].score
        tagged = _mark_reranked(results)
        assert tagged[0].score == original_score

    def test_mark_reranked_does_not_mutate_input(self):
        results = _make_results(["a"])
        _mark_reranked(results)
        assert results[0].source != "reranked"

    def test_reranker_fallback_when_no_key(self, monkeypatch):
        """Reranker without API key returns fallback list synchronously."""
        import asyncio

        from backend.retrieval.reranker import Reranker

        monkeypatch.setattr("backend.config.settings.cohere_api_key", "")
        r = Reranker()
        assert not r.available

        candidates = _make_results(["a", "b", "c", "d", "e"])
        top = asyncio.run(r.rerank("query", candidates, top_k=3))
        assert len(top) == 3
        assert all(t.source == "reranked" for t in top)
