# DentaBot — BrightSmile Dental Clinic Voice & Chat Assistant

A fully local, CPU-friendly conversational AI system for dental clinic appointment scheduling — built across four phases: prompt-orchestrated dialogue, a FastAPI/WebSocket microservice, a local voice pipeline (ASR + TTS), and RAG + tool-calling (CRM, appointments, weather, cost lookup).

## Features

- **Conversation Manager** — finite-state dialogue flow (greeting → intent detection → booking → confirmation) with slot memory, importance-based context filtering, and policy guardrails (no diagnoses, no CNIC/payment collection, emergency escalation).
- **Local LLM backend** — optional `llama-cpp-python` GGUF inference, with a deterministic rule-based fallback when no model is configured.
- **Streaming API** — FastAPI + WebSocket (`/ws/chat`) with token-by-token streaming, plus an HTTP fallback (`/v1/chat`).
- **Voice pipeline** — local ASR (`faster-whisper`) and TTS (`piper-tts`) via `POST /v1/voice/reply`.
- **RAG** — Chroma + `sentence-transformers` (`all-MiniLM-L6-v2`) over a clinic knowledge base (FAQs, policies, medical terms), chunked with overlap and indexed offline.
- **Tools** (SQLite-backed where relevant):
  - `CRMTool` — patient profile storage/retrieval by session
  - `AppointmentTool` — slot lookup, booking, rescheduling, cancellation
  - `WeatherTool` — live weather via wttr.in
  - `DentalCostTool` — local PKR cost estimates for procedures
- **Web UI** — ChatGPT-style chat interface with streaming, history, and session reset.
- **Dockerized** deployment with `docker-compose`.

## Project Structure
app/
├── main.py            # FastAPI routes + WebSocket endpoint
├── models.py           # Pydantic schemas / event types
├── engine.py            # Conversation manager, state machine, prompt templates, tool orchestrator
├── RAG.py               # Embedding + Chroma vector store + retrieval
├── tools.py             # CRM, Appointment, Weather, Dental Cost tools (SQLite)
├── voice.py             # Local ASR (faster-whisper) + TTS (piper) pipeline
├── downRAGData.py       # Script to pull/build the RAG document collection
├── documents/           # Knowledge base source files for RAG indexing
└── web/                 # Static chat UI (HTML/CSS/JS)
Dockerfile
docker-compose.yml
requirements.txt
requirements-llm.txt
dentabot.db              # SQLite DB (patients + appointments)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000` for the chat UI.

### Enable local LLM inference
```bash
pip install -r requirements-llm.txt
export DENTABOT_MODEL_PATH=/path/to/model.gguf
export DENTABOT_N_CTX=2048
export DENTABOT_GPU_LAYERS=-1
```
`GET /healthz` reports `"backend": "llama-cpp"` once active (otherwise `"rule-fallback"`).

### Enable voice
```bash
export DENTABOT_ASR_MODEL=tiny.en
export DENTABOT_ASR_DEVICE=cpu
export DENTABOT_ASR_COMPUTE_TYPE=int8
export DENTABOT_TTS_MODEL_PATH=/path/to/piper/en_US-lessac-medium.onnx
export DENTABOT_TTS_CONFIG_PATH=/path/to/piper/en_US-lessac-medium.onnx.json
export DENTABOT_VOICE_CONCURRENCY=4
```

### Build the RAG index
```bash
python app/downRAGData.py     # fetch source data into app/documents
```
The vector store auto-indexes on first run if empty (Chroma persisted to `./vector_db`).

### Docker
```bash
docker compose up --build
```

## API

### `WS /ws/chat`
```json
{"type":"chat","session_id":"demo-1","message":"I need an appointment","stream":true}
```
Event sequence: `ack` → `start` → `token`* → `complete`. Also supports `{"type":"reset"}` and `{"type":"ping"}`.

### `POST /v1/chat`
```json
{"session_id":"demo-1","message":"What are your hours?"}
```

### `POST /v1/voice/reply` (multipart)
Fields: `session_id`, `audio_extension` (`webm`/`wav`), `audio` (binary). Returns transcript, reply, timing breakdown (ASR/LLM/TTS ms), and base64 WAV reply audio.

### `GET /healthz`
Reports LLM backend, ASR/TTS readiness, and init errors.

## Known Limitations

- Rule-based fallback is used when no GGUF model path is configured — response quality depends on having a real model wired up.
- SQLite is single-file/local; not suitable for multi-instance deployment without a shared DB.
- Voice pipeline requires model files (Whisper + Piper) to be downloaded separately; not bundled in the repo.
- Weather tool depends on the free wttr.in endpoint (no key, no SLA).

## Environment Variables Summary

| Variable | Purpose |
|---|---|
| `DENTABOT_MODEL_PATH` | Path to GGUF LLM |
| `DENTABOT_N_CTX`, `DENTABOT_GPU_LAYERS` | LLM inference tuning |
| `DENTABOT_ASR_MODEL/DEVICE/COMPUTE_TYPE` | faster-whisper config |
| `DENTABOT_TTS_MODEL_PATH`, `DENTABOT_TTS_CONFIG_PATH` | Piper voice files |
| `DENTABOT_VOICE_CONCURRENCY` | Max concurrent voice requests |
| `DENTABOT_DB_PATH` | SQLite path (default `dentabot.db`) |
