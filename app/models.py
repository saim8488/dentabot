from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClientMessageType(str, Enum):
    chat = "chat"
    reset = "reset"
    ping = "ping"


class WSChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ClientMessageType = ClientMessageType.chat
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    message: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    stream: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HTTPChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


class HTTPChatResponse(BaseModel):
    session_id: str
    reply: str
    state: str
    turn_count: int


class HTTPVoiceResponse(BaseModel):
    session_id: str
    transcript: str
    reply: str
    state: str
    turn_count: int
    asr_ms: int
    llm_ms: int
    tts_ms: int
    total_ms: int
    audio_base64: Optional[str] = None


class SessionState(str, Enum):
    greeting = "GREETING"
    intent = "INTENT_DETECTION"
    booking = "BOOKING"
    faq = "FAQ"
    emergency = "EMERGENCY"
    fallback = "FALLBACK"


class EventType(str, Enum):
    ack = "ack"
    start = "start"
    token = "token"
    complete = "complete"
    error = "error"
    info = "info"
    pong = "pong"
