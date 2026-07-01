from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from faster_whisper import WhisperModel
    _FW_IMPORT_ERROR = None
except Exception:  # pragma: no cover
    _FW_IMPORT_ERROR = __import__("traceback").format_exc()
    WhisperModel = None  # type: ignore[assignment]

try:
    from piper.voice import PiperVoice
    _PIPER_IMPORT_ERROR = None
except Exception:  # pragma: no cover
    _PIPER_IMPORT_ERROR = __import__("traceback").format_exc()
    PiperVoice = None  # type: ignore[assignment]


@dataclass
class VoiceReply:
    transcript: str
    reply: str
    audio_wav: Optional[bytes]
    asr_ms: int
    llm_ms: int
    tts_ms: int
    total_ms: int


class LocalASR:
    def __init__(self) -> None:
        self.model_name = os.getenv("DENTABOT_ASR_MODEL", "tiny.en").strip() or "tiny.en"
        self.device = os.getenv("DENTABOT_ASR_DEVICE", "cpu").strip() or "cpu"
        self.compute_type = os.getenv("DENTABOT_ASR_COMPUTE_TYPE", "int8").strip() or "int8"
        self.language = os.getenv("DENTABOT_ASR_LANGUAGE", "en").strip() or "en"
        self.cpu_threads = int(os.getenv("DENTABOT_ASR_CPU_THREADS", str(os.cpu_count() or 4)))

        self.model = None
        self.init_error: Optional[str] = None
        self.enabled = WhisperModel is not None

        if not self.enabled:
            self.init_error = "faster-whisper import failed"
            if _FW_IMPORT_ERROR:
                self.init_error += f": {_FW_IMPORT_ERROR.splitlines()[-1]}"
            return

        try:
            self.model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads
            )
        except Exception as exc:  # pragma: no cover
            self.init_error = str(exc)
            self.enabled = False

    def transcribe(self, audio_bytes: bytes, suffix: str = ".webm") -> str:
        if not self.enabled or self.model is None:
            raise RuntimeError(f"ASR is unavailable: {self.init_error or 'model not loaded'}")

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            segments, _ = self.model.transcribe(
                temp_path,
                language=self.language,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            text_parts = [s.text.strip() for s in segments if s.text.strip()]
            return " ".join(text_parts).strip()
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


class LocalTTS:
    def __init__(self) -> None:
        self.model_path = os.getenv("DENTABOT_TTS_MODEL_PATH", "").strip()
        self.config_path = os.getenv("DENTABOT_TTS_CONFIG_PATH", "").strip()

        self.voice = None
        self.init_error: Optional[str] = None
        self.enabled = PiperVoice is not None and bool(self.model_path)

        if not self.enabled:
            if PiperVoice is None:
                self.init_error = "piper-tts import failed"
                if _PIPER_IMPORT_ERROR:
                    self.init_error += f": {_PIPER_IMPORT_ERROR.splitlines()[-1]}"
            else:
                self.init_error = "DENTABOT_TTS_MODEL_PATH is not set"
            return

        model = Path(self.model_path)
        if not model.exists():
            self.enabled = False
            self.init_error = f"TTS model not found: {model}"
            return

        cfg = Path(self.config_path) if self.config_path else model.with_suffix(model.suffix + ".json")
        if not cfg.exists():
            self.enabled = False
            self.init_error = f"TTS config not found: {cfg}"
            return

        try:
            self.voice = PiperVoice.load(str(model), config_path=str(cfg))
        except Exception as exc:  # pragma: no cover
            self.init_error = str(exc)
            self.enabled = False

    def synthesize(self, text: str) -> bytes:
        if not self.enabled or self.voice is None:
            raise RuntimeError(f"TTS is unavailable: {self.init_error or 'voice not loaded'}")

        # Piper can emit zero chunks on empty/invalid text. Ensure non-empty synthesis input.
        safe_text = (text or "").strip() or "Okay."
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(safe_text, wav_file)
        return wav_buffer.getvalue()


class VoicePipeline:
    def __init__(self, engine_chat_callable) -> None:
        self._engine_chat = engine_chat_callable
        self.asr = LocalASR()
        self.tts = LocalTTS()

    @property
    def asr_ready(self) -> bool:
        return self.asr.enabled

    @property
    def tts_ready(self) -> bool:
        return self.tts.enabled

    async def run(self, session, audio_bytes: bytes, audio_extension: str = "webm") -> VoiceReply:
        started = time.perf_counter()
        safe_ext = "".join(ch for ch in audio_extension.lower() if ch.isalnum()) or "webm"

        asr_t0 = time.perf_counter()
        transcript = await asyncio.to_thread(self.asr.transcribe, audio_bytes, f".{safe_ext}")
        asr_ms = int((time.perf_counter() - asr_t0) * 1000)
        if not transcript:
            transcript = "I could not hear clear speech. Please ask the patient to repeat."

        llm_t0 = time.perf_counter()
        reply = await self._engine_chat(session, transcript)
        llm_ms = int((time.perf_counter() - llm_t0) * 1000)

        tts_ms = 0
        audio_wav: Optional[bytes] = None
        if self.tts_ready:
            tts_t0 = time.perf_counter()
            audio_wav = await asyncio.to_thread(self.tts.synthesize, reply)
            tts_ms = int((time.perf_counter() - tts_t0) * 1000)

        total_ms = int((time.perf_counter() - started) * 1000)
        return VoiceReply(
            transcript=transcript,
            reply=reply,
            audio_wav=audio_wav,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
        )
