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
