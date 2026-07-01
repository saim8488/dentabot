from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import DentaBotEngine, SessionStore
from .models import EventType, HTTPChatRequest, HTTPChatResponse, HTTPVoiceResponse, WSChatRequest
from .voice import VoicePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dentabot.phase4")

app = FastAPI(title="DentaBot Phase IV", version="1.0.0")
store = SessionStore()
engine = DentaBotEngine()
voice = VoicePipeline(engine.chat)
voice_semaphore = asyncio.Semaphore(int(os.getenv("DENTABOT_VOICE_CONCURRENCY", "4")))
WEB_ROOT = Path(__file__).resolve().parent / "web"
ASSETS_ROOT = WEB_ROOT / "assets"

app.mount("/assets", StaticFiles(directory=ASSETS_ROOT), name="assets")


@app.middleware("http")
async def no_cache_for_ui(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _event(event_type: EventType, session_id: str, **data):
    return {
        "type": event_type.value,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "dentabot-phase4",
            "backend": engine.backend,
            "asr_ready": voice.asr_ready,
            "tts_ready": voice.tts_ready,
            "asr_error": voice.asr.init_error,
            "tts_error": voice.tts.init_error,
        }
    )


@app.get("/", response_class=FileResponse)
async def chat_ui() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.post("/v1/chat", response_model=HTTPChatResponse)
async def chat_http(payload: HTTPChatRequest) -> HTTPChatResponse:
    session = await store.get_or_create(payload.session_id)
    async with session.lock:
        reply = await engine.chat(session, payload.message)
        return HTTPChatResponse(
            session_id=session.session_id,
            reply=reply,
            state=session.state.value,
            turn_count=session.turn_count,
        )


@app.post("/v1/voice/reply", response_model=HTTPVoiceResponse)
async def voice_reply(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
    audio_extension: str = Form(default="webm"),
) -> HTTPVoiceResponse:
    if not voice.asr_ready:
        return HTTPVoiceResponse(
            session_id=session_id,
            transcript="",
            reply=f"ASR not ready: {voice.asr.init_error or 'unknown error'}",
            state="FALLBACK",
            turn_count=0,
            asr_ms=0,
            llm_ms=0,
            tts_ms=0,
            total_ms=0,
            audio_base64=None,
        )

    session = await store.get_or_create(session_id)
    audio_bytes = await audio.read()
    if not audio_bytes:
        return HTTPVoiceResponse(
            session_id=session_id,
            transcript="",
            reply="No audio received.",
            state=session.state.value,
            turn_count=session.turn_count,
            asr_ms=0,
            llm_ms=0,
            tts_ms=0,
            total_ms=0,
            audio_base64=None,
        )

    async with voice_semaphore:
        async with session.lock:
            result = await voice.run(session, audio_bytes=audio_bytes, audio_extension=audio_extension)

    audio_base64 = base64.b64encode(result.audio_wav).decode("ascii") if result.audio_wav else None
    return HTTPVoiceResponse(
        session_id=session.session_id,
        transcript=result.transcript,
        reply=result.reply,
        state=session.state.value,
        turn_count=session.turn_count,
        asr_ms=result.asr_ms,
        llm_ms=result.llm_ms,
        tts_ms=result.tts_ms,
        total_ms=result.total_ms,
        audio_base64=audio_base64,
    )


@app.post("/v1/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    audio_extension: str = Form(default="webm"),
) -> JSONResponse:
    if not voice.asr_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": voice.asr.init_error or "ASR not ready"},
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        return JSONResponse(status_code=400, content={"status": "error", "detail": "No audio received."})

    try:
        started = datetime.now(timezone.utc)
        safe_ext = "".join(ch for ch in audio_extension.lower() if ch.isalnum()) or "webm"
        text = await asyncio.to_thread(voice.asr.transcribe, audio_bytes, f".{safe_ext}")
        elapsed_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        return JSONResponse({"status": "ok", "transcript": text, "asr_ms": elapsed_ms})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(exc)})


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    default_session_id = f"ws-{uuid4().hex[:12]}"

    await websocket.send_json(
        _event(
            EventType.info,
            default_session_id,
            message="connected",
            accepted_types=["chat", "reset", "ping"],
        )
    )

    while True:
        try:
            raw = await websocket.receive_text()
            try:
                parsed = json.loads(raw)
                req = WSChatRequest.model_validate(parsed)
            except Exception as exc:
                await websocket.send_json(
                    _event(EventType.error, default_session_id, code="bad_request", detail=str(exc))
                )
                continue

            session_id = req.session_id or default_session_id
            session = await store.get_or_create(session_id)

            if req.type.value == "ping":
                await websocket.send_json(_event(EventType.pong, session_id, message="pong"))
                continue

            if req.type.value == "reset":
                async with session.lock:
                    session.reset()
                    engine.reset_session(session_id)
                await websocket.send_json(_event(EventType.info, session_id, message="session_reset"))
                continue

            if not req.message:
                await websocket.send_json(
                    _event(EventType.error, session_id, code="missing_message", detail="message is required")
                )
                continue

            await websocket.send_json(_event(EventType.ack, session_id, stream=req.stream))

            async with session.lock:
                reply = await engine.chat(session, req.message)
                state = session.state.value
                turn_count = session.turn_count

            if req.stream:
                await websocket.send_json(_event(EventType.start, session_id, state=state, turn_count=turn_count))
                async for token in engine.stream_tokens(reply):
                    await websocket.send_json(_event(EventType.token, session_id, token=token))

            await websocket.send_json(
                _event(
                    EventType.complete,
                    session_id,
                    reply=reply,
                    state=state,
                    turn_count=turn_count,
                )
            )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            return
        except Exception as exc:
            logger.exception("Unhandled websocket error")
            try:
                await websocket.send_json(
                    _event(EventType.error, default_session_id, code="server_error", detail=str(exc))
                )
            except Exception:
                return
