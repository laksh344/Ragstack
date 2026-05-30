"""Conversation and message models for the chat API.

Messages are stored in Redis as JSON-serialised Message objects.
Conversations are logical groupings identified by a UUID that the
client generates or receives on first contact.
"""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.agent.state import Citation


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # "user" | "assistant"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    messages: list[Message] = Field(default_factory=list)
    message_count: int = 0
