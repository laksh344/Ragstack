"""Unit tests for the chat API layer.

All tests are pure Python — no Redis, no LLM, no graph invocation.
FastAPI route logic is tested by mocking the graph and Redis client.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.chat import (
    _MAX_HISTORY_MESSAGES,
    ChatRequest,
    FeedbackRequest,
    _sse,
    _to_langchain_messages,
)
from backend.models.conversation import Conversation, Message

# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestMessage:
    def test_defaults_populated(self):
        m = Message(role="user", content="hello")
        assert m.id  # uuid auto-generated
        assert m.timestamp
        assert m.citations == []
        assert m.guardrail_flags == []

    def test_role_preserved(self):
        m = Message(role="assistant", content="hi there")
        assert m.role == "assistant"

    def test_serialise_deserialise(self):
        m = Message(role="user", content="What is RAG?")
        raw = m.model_dump_json()
        restored = Message.model_validate(json.loads(raw))
        assert restored.id == m.id
        assert restored.content == m.content
        assert restored.role == m.role


class TestConversation:
    def test_defaults(self):
        c = Conversation()
        assert c.id
        assert c.created_at
        assert c.messages == []
        assert c.message_count == 0

    def test_with_messages(self):
        msgs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        c = Conversation(id="test-id", messages=msgs, message_count=2)
        assert len(c.messages) == 2
        assert c.message_count == 2


# ---------------------------------------------------------------------------
# ChatRequest validation
# ---------------------------------------------------------------------------


class TestChatRequest:
    def test_query_required(self):
        r = ChatRequest(query="hello")
        assert r.query == "hello"
        assert r.conversation_id is None
        assert r.dataset_id is None

    def test_conversation_id_optional(self):
        r = ChatRequest(query="hi", conversation_id="abc-123")
        assert r.conversation_id == "abc-123"

    def test_dataset_id_optional(self):
        r = ChatRequest(query="hi", dataset_id="my-dataset")
        assert r.dataset_id == "my-dataset"


class TestFeedbackRequest:
    def test_required_fields(self):
        r = FeedbackRequest(message_id="msg-1", rating=1)
        assert r.message_id == "msg-1"
        assert r.rating == 1
        assert r.comment == ""

    def test_negative_rating(self):
        r = FeedbackRequest(message_id="msg-2", rating=-1, comment="wrong answer")
        assert r.rating == -1
        assert r.comment == "wrong answer"


# ---------------------------------------------------------------------------
# SSE formatting
# ---------------------------------------------------------------------------


class TestSSE:
    def test_format_token_event(self):
        line = _sse({"type": "token", "content": "hello "})
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        parsed = json.loads(line[len("data: "):].strip())
        assert parsed["type"] == "token"
        assert parsed["content"] == "hello "

    def test_format_done_event(self):
        line = _sse({"type": "done", "message_id": "x", "conversation_id": "y"})
        parsed = json.loads(line[len("data: "):].strip())
        assert parsed["type"] == "done"
        assert parsed["message_id"] == "x"

    def test_format_error_event(self):
        line = _sse({"type": "error", "detail": "something went wrong"})
        parsed = json.loads(line[len("data: "):].strip())
        assert parsed["detail"] == "something went wrong"

    def test_double_newline_terminator(self):
        line = _sse({"type": "token", "content": "x"})
        assert line[-2:] == "\n\n"


# ---------------------------------------------------------------------------
# to_langchain_messages
# ---------------------------------------------------------------------------


class TestToLangchainMessages:
    def test_user_becomes_human_message(self):
        from langchain_core.messages import HumanMessage

        history = [Message(role="user", content="What is RAG?")]
        msgs = _to_langchain_messages(history)
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "What is RAG?"

    def test_assistant_becomes_ai_message(self):
        from langchain_core.messages import AIMessage

        history = [Message(role="assistant", content="RAG is...")]
        msgs = _to_langchain_messages(history)
        assert isinstance(msgs[0], AIMessage)

    def test_mixed_history(self):
        from langchain_core.messages import AIMessage, HumanMessage

        history = [
            Message(role="user", content="Q"),
            Message(role="assistant", content="A"),
            Message(role="user", content="Q2"),
        ]
        msgs = _to_langchain_messages(history)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)

    def test_unknown_role_skipped(self):
        history = [Message(role="system", content="ignored")]
        msgs = _to_langchain_messages(history)
        assert msgs == []

    def test_empty_history(self):
        assert _to_langchain_messages([]) == []


# ---------------------------------------------------------------------------
# FastAPI route smoke tests (mocked graph + mocked Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from backend.main import app
    return TestClient(app, raise_server_exceptions=False)


class TestChatEndpoint:
    def test_returns_streaming_content_type(self, client):
        fake_result = {
            "response": "The answer is 42.",
            "citations": [],
            "guardrail_flags": [],
        }
        with (
            patch("backend.api.chat._get_redis", return_value=AsyncMock(return_value=None)),
            patch("backend.agent.graph.graph") as mock_graph,
        ):
            mock_graph.ainvoke = AsyncMock(return_value=fake_result)
            response = client.post(
                "/api/v1/chat",
                json={"query": "What is the answer?"},
                headers={"Accept": "text/event-stream"},
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_streaming_body_contains_done_event(self, client):
        fake_result = {
            "response": "Hello world.",
            "citations": [],
            "guardrail_flags": [],
        }
        with (
            patch("backend.api.chat._get_redis", return_value=AsyncMock(return_value=None)),
            patch("backend.agent.graph.graph") as mock_graph,
        ):
            mock_graph.ainvoke = AsyncMock(return_value=fake_result)
            response = client.post("/api/v1/chat", json={"query": "Hi"})

        body = response.text
        events = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        event_types = [e["type"] for e in events]
        assert "done" in event_types
        assert "token" in event_types

    def test_feedback_endpoint_no_redis(self, client):
        with patch("backend.api.chat._get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = None
            response = client.post(
                "/api/v1/chat/feedback",
                json={"message_id": "msg-abc", "rating": 1},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["stored"] is False

    def test_get_conversation_no_redis_returns_503(self, client):
        with patch("backend.api.chat._get_redis", new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = None
            response = client.get("/api/v1/chat/some-conv-id")
        assert response.status_code == 503

    def test_get_conversation_not_found_returns_404(self, client):
        mock_redis = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])
        mock_redis.aclose = AsyncMock()
        with patch("backend.api.chat._get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_redis
            response = client.get("/api/v1/chat/nonexistent-id")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Conversation history max-length guard
# ---------------------------------------------------------------------------


class TestHistoryLimit:
    def test_max_history_constant_positive(self):
        assert _MAX_HISTORY_MESSAGES > 0

    def test_message_serialisation_round_trip(self):
        """All messages used in history must survive JSON round-trips."""
        msgs = [
            Message(role="user", content=f"question {i}",
                    citations=[], guardrail_flags=[])
            for i in range(_MAX_HISTORY_MESSAGES + 5)
        ]
        for m in msgs:
            restored = Message.model_validate(json.loads(m.model_dump_json()))
            assert restored.id == m.id
