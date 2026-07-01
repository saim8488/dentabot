from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from .RAG import retrieve_relevant_chunks
from .models import SessionState
from .tools import AppointmentTool, CRMTool, DentalCostTool, WeatherTool

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None  # type: ignore[assignment]


BASE_SYSTEM_PROMPT = """You are DentaBot, the virtual receptionist for BrightSmile Dental Clinic.
Your sole purpose is to help patients with appointment scheduling and general clinic questions.

CLINIC INFO:
  Name    : BrightSmile Dental Clinic
  Hours   : Monday-Saturday, 9:00 AM - 7:00 PM
  Location: 42 Gulberg III, Lahore
  Hotline : 042-35001234
  Insurance accepted: Jubilee Health, EFU, Adamjee, State Life

SERVICES OFFERED:
  Routine check-ups, teeth cleaning, braces, Invisalign, teeth whitening,
  tooth extractions, fillings, root canals, dental X-rays.

AVAILABLE DOCTORS:
  Dr. Rehan  (General Dentistry) - Mon/Wed/Fri
  Dr. Nadia  (General + Cosmetic) - Tue/Thu/Sat
  Dr. Farhan (Orthodontics)       - Mon/Thu/Sat

TONE & STYLE:
  - Warm, professional, and concise (2-4 sentences per response).
  - Address patients by first name once you have it.
  - Never use medical jargon or slang.
  - Always end with a clear next question or action.

HARD LIMITS - NEVER violate these:
  - Do NOT diagnose, prescribe, or give clinical opinions.
  - Do NOT ask for CNIC, medical records, or payment details.
  - Do NOT speculate on insurance coverage; advise calling the clinic for details.
  - For emergencies (severe pain, swelling, bleeding), offer the soonest slot AND give hotline: 042-35001234.""".strip()


STATE_PROMPTS = {
    "GREETING": """
CURRENT TASK - GREETING:
Greet the patient warmly. Ask whether they are a new or existing patient.
Do not ask for any other information yet.""",
    "INTENT_DETECTION": """
CURRENT TASK - UNDERSTAND INTENT:
Identify what the patient needs. Possible intents:
  1. Book a new appointment
  2. Reschedule an existing appointment
  3. Cancel an appointment
  4. Ask a question (hours, services, insurance, location)
If unclear after one response, present a numbered menu of these options.""",
    "BOOKING_NAME": """
CURRENT TASK - COLLECT NAME:
You are booking an appointment. Ask for the patient's full name.
Ask for name only - nothing else yet.""",
    "BOOKING_CONTACT": """
CURRENT TASK - COLLECT CONTACT:
You have the patient's name. Now ask for their contact number.
Ask for contact only - nothing else yet.""",
    "BOOKING_SERVICE": """
CURRENT TASK - COLLECT SERVICE TYPE:
You have name and contact. Ask what type of appointment they need.
Give 2-3 examples: routine check-up, teeth cleaning, specific concern, cosmetic consultation.""",
    "BOOKING_DATETIME": """
CURRENT TASK - COLLECT DATE & TIME:
Use the exact TODAY and TOMORROW dates from the system clock above.
Suggest 3 specific upcoming slots with real day names and dates.
If the patient gives a date that is in the past, politely say that date has
already passed and ask them to choose a future date.
Do NOT book anything yet - offer options and wait for the patient to choose.
Doctor schedule: Dr. Rehan (Mon/Wed/Fri), Dr. Nadia (Tue/Thu/Sat), Dr. Farhan Ortho (Mon/Thu/Sat)""",
    "BOOKING_CONFIRM": """
CURRENT TASK - CONFIRM BOOKING:
Read back the full booking summary: name, service, date, time, doctor.
Ask the patient to confirm. Do not finalize until they say yes.""",
    "BOOKING_DONE": """
CURRENT TASK - BOOKING COMPLETE:
The patient confirmed. Announce the booking is confirmed. Mention they will
receive a reminder. Ask if there is anything else you can help with.""",
    "RESCHEDULE": """
CURRENT TASK - RESCHEDULE:
Ask for the patient's name and the date of their current appointment to look it up.
Then offer 2-3 alternative slots. Confirm the new slot before finalizing.""",
    "CANCEL": """
CURRENT TASK - CANCEL APPOINTMENT:
Ask for the patient's name and appointment date to locate the booking.
Confirm they want to cancel. Once confirmed, acknowledge cancellation warmly
and offer to rebook in the future.""",
    "FAQ": """
CURRENT TASK - ANSWER QUESTION:
Answer the patient's question using only the clinic information you have been given.
Be concise. After answering, ask if there is anything else you can help with.""",
    "EMERGENCY": """
CURRENT TASK - EMERGENCY:
The patient may be experiencing a dental emergency.
Express empathy. Immediately offer the earliest available appointment slot.
Provide the clinic hotline: 042-35001234.
Advise them to go to the emergency room if the situation is life-threatening.""",
    "FALLBACK": """
CURRENT TASK - UNCLEAR INPUT:
The patient's message was unclear or outside your scope.
Politely say you can only help with appointment scheduling and clinic questions.
Present a short numbered menu of what you can do.""",
    "CLOSING": """
CURRENT TASK - CLOSING:
The patient is done. Wish them well with a warm closing message.
Remind them of the clinic hotline if they need anything else.""",
}


class State(Enum):
    GREETING = auto()
    INTENT_DETECTION = auto()
    BOOKING_NAME = auto()
    BOOKING_CONTACT = auto()
    BOOKING_SERVICE = auto()
    BOOKING_DATETIME = auto()
    BOOKING_CONFIRM = auto()
    BOOKING_DONE = auto()
    RESCHEDULE = auto()
    CANCEL = auto()
    FAQ = auto()
    EMERGENCY = auto()
    FALLBACK = auto()
    CLOSING = auto()
    END = auto()


@dataclass
class AppointmentSlots:
    patient_status: Optional[str] = None
    patient_name: Optional[str] = None
    contact_number: Optional[str] = None
    service_type: Optional[str] = None
    preferred_date: Optional[str] = None
    preferred_time: Optional[str] = None
    doctor: Optional[str] = None
    intent: Optional[str] = None

    def to_context_string(self) -> str:
        filled = {k: v for k, v in self.__dict__.items() if v}
        if not filled:
            return ""
        lines = ["[Collected information]"]
        for k, v in filled.items():
            lines.append(f"  {k.replace('_', ' ').title()}: {v}")
        return "\n".join(lines)


SIGNAL_PATTERNS = [
    r"\b(name|call me|i am|i\'m)\b",
    r"\b(\d{4}[\-\s]?\d{7,8}|0\d{9,10})\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(\d{1,2}(:\d{2})?\s?(am|pm))\b",
    r"\b(today|tomorrow|next week)\b",
    r"\b(checkup|check.up|cleaning|braces|whitening|extraction|filling|consultation)\b",
    r"\b(book|schedule|reschedule|cancel|appointment)\b",
    r"\b(yes|confirmed|correct|that works|sounds good)\b",
]


def importance_score(text: str) -> int:
    score = 0
    lower = text.lower()
    for pat in SIGNAL_PATTERNS:
        if re.search(pat, lower):
            score += 1
    return score


def parse_date(text: str) -> Optional[datetime]:
    lower = text.lower()
    now = datetime.now()

    if "today" in lower:
        return now
    if "tomorrow" in lower:
        return now + timedelta(days=1)

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(days):
        if day in lower:
            diff = (i - now.weekday()) % 7
            if diff == 0:
                diff = 7
            return now + timedelta(days=diff)

    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }

    for name, month_num in months.items():
        if name in lower:
            day_m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", lower)
            year_m = re.search(r"\b(20\d{2})\b", lower)
            day = int(day_m.group(1)) if day_m else 1
            year = int(year_m.group(1)) if year_m else now.year
            try:
                return datetime(year, month_num, day)
            except ValueError:
                return None
    return None


def detect_intent(text: str) -> str:
    lower = text.lower().strip()

    def matches(words: List[str]) -> bool:
        return any(re.search(r"\b" + re.escape(w) + r"\b", lower) for w in words)

    if re.fullmatch(r"(?:i am|i'm)?\s*(?:a\s*)?(?:new|existing)\s*patient[.!?]?", lower):
        return "patient_status"
    if matches(["new patient", "existing patient"]) and len(lower.split()) <= 5:
        return "patient_status"
    if matches(["what did i tell you", "what did i say", "do you remember", "summarize our chat", "recap", "what have i told you"]):
        return "memory_query"
    if matches(["hello", "hi", "hey", "good morning", "good evening", "how are you", "thanks", "thank you"]):
        return "chit_chat"

    if matches(["toothache", "pain", "ache", "hurts", "swollen", "swelling", "bleeding", "emergency", "severe", "broken tooth", "abscess"]):
        return "emergency"
    # Catch service-name-as-booking: "i need a check", "need a cleaning", "want braces", etc.
    booking_services = [
        "book", "schedule", "appointment", "see a dentist", "make an appointment",
        "check-up", "checkup", "check up", "cleaning", "braces", "invisalign",
        "whitening", "extraction", "filling", "root canal", "x-ray", "consultation",
        "need a check", "need a cleaning", "need a filling", "want an appointment",
        "i need", "i want", "i'd like", "can i get", "can i book", "fix my",
    ]
    if matches(booking_services):
        return "book"
    if matches(["reschedule", "change my appointment", "move my booking"]):
        return "reschedule"
    if matches(["cancel", "remove my appointment"]):
        return "cancel"
    if matches(["hours", "open", "location", "where", "insurance", "cover", "accept", "services", "offer", "how much", "price", "cost", "am i a new patient", "am i an existing patient"]):
        return "faq"
    if matches(["yes", "yeah", "yep", "correct", "confirm", "okay", "that works", "sounds good", "perfect", "go ahead"]):
        return "affirm"
    if matches(["no", "nope", "nothing else", "bye", "goodbye", "never mind", "that's all"]):
        return "deny"
    return "unknown"


def next_state(current: State, intent: str, slots: AppointmentSlots) -> State:
    if current == State.GREETING:
        if intent == "emergency":
            return State.EMERGENCY
        return State.INTENT_DETECTION

    if current == State.INTENT_DETECTION:
        if intent == "patient_status":
            return State.INTENT_DETECTION
        if intent in ("memory_query", "chit_chat"):
            return State.INTENT_DETECTION
        if intent == "book":
            return State.BOOKING_NAME
        if intent == "reschedule":
            return State.RESCHEDULE
        if intent == "cancel":
            return State.CANCEL
        if intent == "faq":
            return State.FAQ
        if intent == "emergency":
            return State.EMERGENCY
        if slots.intent == "book":
            if not slots.patient_name:
                return State.BOOKING_NAME
            if not slots.contact_number:
                return State.BOOKING_CONTACT
            if not slots.service_type:
                return State.BOOKING_SERVICE
            if not slots.preferred_date or not slots.preferred_time:
                return State.BOOKING_DATETIME
            return State.BOOKING_CONFIRM
        return State.FALLBACK

    if current in (State.BOOKING_NAME, State.BOOKING_CONTACT, State.BOOKING_SERVICE, State.BOOKING_DATETIME, State.BOOKING_CONFIRM):
        if intent == "emergency":
            return State.EMERGENCY
        if not slots.patient_name:
            return State.BOOKING_NAME
        if not slots.contact_number:
            return State.BOOKING_CONTACT
        if not slots.service_type:
            return State.BOOKING_SERVICE
        if not slots.preferred_date or not slots.preferred_time:
            return State.BOOKING_DATETIME
        if current != State.BOOKING_CONFIRM:
            return State.BOOKING_CONFIRM
        if intent == "affirm":
            return State.BOOKING_DONE
        if intent == "deny":
            return State.BOOKING_DATETIME
        return State.BOOKING_CONFIRM

    if current == State.BOOKING_DONE:
        if intent in ("book", "reschedule", "cancel", "faq"):
            return State.INTENT_DETECTION
        return State.CLOSING

    if current == State.RESCHEDULE:
        if intent == "affirm":
            return State.BOOKING_DONE
        return State.RESCHEDULE

    if current == State.CANCEL:
        if intent in ("affirm", "deny"):
            return State.CLOSING
        return State.CANCEL

    if current == State.FAQ:
        if intent in ("memory_query", "chit_chat"):
            return State.INTENT_DETECTION
        if intent == "book":
            return State.BOOKING_NAME
        if intent == "reschedule":
            return State.RESCHEDULE
        if intent == "cancel":
            return State.CANCEL
        if intent == "deny":
            return State.CLOSING
        return State.FAQ

    if current == State.EMERGENCY:
        if intent == "affirm":
            return State.BOOKING_CONFIRM
        return State.EMERGENCY

    if current == State.FALLBACK:
        return State.INTENT_DETECTION

    if current == State.CLOSING:
        return State.END

    return State.INTENT_DETECTION


def _api_state(state: State) -> SessionState:
    if state in {State.FAQ}:
        return SessionState.faq
    if state in {State.EMERGENCY}:
        return SessionState.emergency
    if state in {State.BOOKING_NAME, State.BOOKING_CONTACT, State.BOOKING_SERVICE,
                 State.BOOKING_DATETIME, State.BOOKING_CONFIRM, State.BOOKING_DONE,
                 State.RESCHEDULE, State.CANCEL}:
        return SessionState.booking
    if state in {State.INTENT_DETECTION, State.GREETING}:
        return SessionState.intent
    return SessionState.fallback


# ══════════════════════════════════════════════════════════════════════════════
# Tool Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class ToolOrchestrator:
    """
    Wraps the four tools and decides which to call based on conversation state.
    All methods are synchronous (called inside asyncio.to_thread).
    Returns a string to inject into the LLM prompt as [TOOL CONTEXT].
    """

    def __init__(self) -> None:
        self.crm = CRMTool()
        self.appointment = AppointmentTool()
        self.weather = WeatherTool()
        self.cost = DentalCostTool()

    # ── CRM helpers ────────────────────────────────────────────────────────

    def load_returning_patient(self, session_id: str) -> str:
        """Called on GREETING — check if this session has been seen before."""
        patient = self.crm.get_patient(session_id)
        if patient and patient.get("name"):
            visits = patient.get("visit_count", 1)
            last_svc = patient.get("last_service") or "a previous appointment"
            return (
                f"[CRM] Returning patient on record: {patient['name']}, "
                f"contact {patient.get('contact', 'unknown')}, "
                f"last service: {last_svc}, total visits: {visits}. "
                "Greet them by name warmly."
            )
        return ""

    def save_patient_info(self, session_id: str, slots: AppointmentSlots) -> None:
        """Persist name/contact/service to CRM whenever we have new data."""
        if slots.patient_name or slots.contact_number:
            self.crm.upsert_patient(
                session_id,
                name=slots.patient_name,
                contact=slots.contact_number,
                last_service=slots.service_type,
            )

    # ── Appointment helpers ────────────────────────────────────────────────

    def get_open_slots_context(self) -> str:
        """Return a formatted list of the next 6 open slots for the prompt."""
        slots = self.appointment.get_available_slots(limit=6)
        if not slots:
            return "[APPOINTMENT SLOTS] No available slots found in the next two weeks."
        lines = ["[AVAILABLE SLOTS — use these exact dates and times when suggesting options]"]
        for s in slots:
            docs = ", ".join(s["doctors"]) if s["doctors"] else "Available Doctor"
            lines.append(f"  • {s['date']} at {s['time']} — {docs}")
        return "\n".join(lines)

    def book_confirmed_appointment(
        self, session_id: str, slots: AppointmentSlots
    ) -> str:
        """Book the appointment once the patient confirms. Returns context string."""
        if not (slots.patient_name and slots.contact_number and
                slots.service_type and slots.preferred_date and slots.preferred_time):
            return ""
        result = self.appointment.book_appointment(
            session_id=session_id,
            patient_name=slots.patient_name,
            contact=slots.contact_number,
            service=slots.service_type,
            date=slots.preferred_date,
            time=slots.preferred_time,
        )
        # Also update CRM with latest service
        self.crm.upsert_patient(session_id, last_service=slots.service_type)

        if result.get("success"):
            return (
                f"[BOOKING CONFIRMED] {result['message']} "
                f"Appointment ID: #{result['appointment_id']}. "
                "Tell the patient their booking is confirmed and give them the ID."
            )
        return f"[BOOKING FAILED] {result.get('message', 'Unknown error')}. Offer an alternative slot."

    # ── Cost helper ────────────────────────────────────────────────────────

    def get_cost_context(self, user_input: str, service_type: Optional[str]) -> str:
        """Return cost info if the patient is asking about pricing."""
        procedure = service_type or user_input
        result = self.cost.get_cost(procedure)
        if "min_cost_pkr" in result:
            return (
                f"[COST INFO] {result['procedure']}: "
                f"PKR {result['min_cost_pkr']:,} – {result['max_cost_pkr']:,}. "
                f"{result['note']}"
            )
        return f"[COST INFO] {result.get('message', '')} {result.get('note', '')}"

    # ── Weather helper ─────────────────────────────────────────────────────

    def get_weather_context(self, user_input: str) -> str:
        """Return weather info for Lahore if the patient asks about it."""
        result = self.weather.get_weather("Lahore")
        if "error" in result:
            return f"[WEATHER] Could not fetch weather: {result['error']}"
        return (
            f"[WEATHER] Lahore right now: {result['description']}, "
            f"{result['temperature_c']}°C (feels like {result['feels_like_c']}°C), "
            f"humidity {result['humidity']}%, wind {result['wind_kmph']} km/h."
        )

    # ── Main dispatch ──────────────────────────────────────────────────────

    def dispatch(
        self,
        session_id: str,
        user_input: str,
        intent: str,
        prev_state: State,
        new_state: State,
        slots: AppointmentSlots,
    ) -> str:
        """
        Decide which tools to call and return combined context string.
        Called synchronously from ConversationManager.chat().
        """
        context_parts: List[str] = []
        lower = user_input.lower()

        # 1. CRM — greet returning patient at session start
        if prev_state == State.GREETING:
            crm_ctx = self.load_returning_patient(session_id)
            if crm_ctx:
                context_parts.append(crm_ctx)

        # 2. CRM — persist name/contact whenever newly collected
        self.save_patient_info(session_id, slots)

        # 3. Appointment — show real open slots when suggesting times
        if new_state == State.BOOKING_DATETIME or prev_state == State.BOOKING_DATETIME:
            context_parts.append(self.get_open_slots_context())

        # 4. Appointment — actually book when patient says yes at confirmation
        if new_state == State.BOOKING_DONE:
            booking_ctx = self.book_confirmed_appointment(session_id, slots)
            if booking_ctx:
                context_parts.append(booking_ctx)

        # 5. Cost tool — patient asks about price/fee/cost
        cost_keywords = ["cost", "price", "how much", "fee", "charge", "rate", "expensive"]
        if any(kw in lower for kw in cost_keywords):
            context_parts.append(self.get_cost_context(user_input, slots.service_type))

        # 6. Weather — patient asks about weather or travel conditions
        weather_keywords = ["weather", "rain", "raining", "sunny", "hot", "cold", "storm", "travel"]
        if any(kw in lower for kw in weather_keywords):
            context_parts.append(self.get_weather_context(user_input))

        return "\n".join(context_parts)


# ══════════════════════════════════════════════════════════════════════════════
# Local LLM Engine (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class LocalLLMEngine:
    def __init__(self) -> None:
        self.model_path = os.getenv("DENTABOT_MODEL_PATH", "").strip()
        file_exists = os.path.isfile(self.model_path) if self.model_path else False
        llama_available = Llama is not None
        self.enabled = bool(self.model_path and file_exists and llama_available)

        print(f"\n[LLM ENGINE] Model path: {self.model_path if self.model_path else 'NOT SET'}")
        print(f"[LLM ENGINE] File exists: {file_exists}")
        print(f"[LLM ENGINE] Llama available: {llama_available}")
        print(f"[LLM ENGINE] Engine enabled: {self.enabled}\n")

        self.n_ctx = int(os.getenv("DENTABOT_N_CTX", "2048"))
        self.n_gpu_layers = int(os.getenv("DENTABOT_GPU_LAYERS", "-1"))
        self.llm = None

        if self.enabled:
            print("[LLM ENGINE] Loading model (this may take 30-60 seconds)...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_threads=os.cpu_count(),
            )
            print("[LLM ENGINE] Model loaded successfully!")

    def count_tokens(self, text: str) -> int:
        if self.llm is None:
            return len(text.split())
        return len(self.llm.tokenize(text.encode("utf-8")))

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 60) -> Dict[str, object]:
        if self.llm is None:
            raise RuntimeError("LLM not configured")
        resp = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stream=False,
        )
        usage = resp.get("usage", {})
        return {
            "content": resp["choices"][0]["message"]["content"].strip(),
            "completion_tokens": usage.get("completion_tokens", 0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Context Memory Manager (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class ContextMemoryManager:
    def __init__(
        self,
        token_counter: Callable[[str], int],
        system_prompt: str,
        max_turns: int = 10,
        token_budget: int = 1800,
    ) -> None:
        self.token_counter = token_counter
        self.base_system = system_prompt
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.history: List[Dict[str, str]] = []
        self.slots = AppointmentSlots()
        self.current_state: Optional[State] = None

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self._extract_slots(content)

    def add_assistant(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self._system_with_slots()}] + self._prune()

    def reset(self) -> None:
        self.history = []
        self.slots = AppointmentSlots()
        self.current_state = None

    def _system_with_slots(self) -> str:
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        date_block = (
            "[SYSTEM CLOCK - DO NOT CONTRADICT THESE VALUES]\n"
            f"  Today    : {now.strftime('%A, %d %B %Y')}\n"
            f"  Time now : {now.strftime('%I:%M %p')}\n"
            f"  Tomorrow : {tomorrow.strftime('%A, %d %B %Y')}\n"
            "These values come from the server clock and are always correct. "
            "If a patient states a different date as 'today', politely correct them. "
            "Never suggest or accept appointment slots in the past. "
            "Never write [insert date] or any placeholder."
        )

        parts: List[str] = []
        slot_ctx = self.slots.to_context_string()
        if slot_ctx:
            parts.append(slot_ctx)
        parts.append(date_block)
        parts.append(self.base_system)
        return "\n\n".join(parts)

    def _prune(self) -> List[Dict[str, str]]:
        pruned = self.history[-(self.max_turns * 2) :]
        budget = self.token_budget - self.token_counter(self._system_with_slots())

        while len(pruned) > 2:
            total = sum(self.token_counter(m["content"]) for m in pruned)
            if total <= budget:
                break
            pairs = [
                (
                    importance_score(pruned[i]["content"])
                    + (importance_score(pruned[i + 1]["content"]) if i + 1 < len(pruned) else 0),
                    i,
                )
                for i in range(0, len(pruned) - 2, 2)
            ]
            if not pairs:
                break
            idx = min(pairs, key=lambda x: x[0])[1]
            pruned = pruned[:idx] + pruned[idx + 2 :]
        return pruned

    def _extract_slots(self, text: str) -> None:
        lower = text.strip().lower()
        state = self.current_state

        if re.search(r"\bnew patient\b", lower):
            self.slots.patient_status = "new"
        elif re.search(r"\bexisting patient\b", lower):
            self.slots.patient_status = "existing"

        if state == State.BOOKING_NAME and not self.slots.patient_name:
            name_candidate = re.sub(r"(my name is|i'm|i am|call me|this is)", "", text, flags=re.IGNORECASE)
            name_candidate = re.sub(r"0\d[\d\s\-]{8,12}", "", name_candidate)
            name_candidate = name_candidate.strip(" .,!?")
            if 1 <= len(name_candidate.split()) <= 4 and name_candidate:
                self.slots.patient_name = name_candidate.title()
            return

        if state == State.BOOKING_CONTACT and not self.slots.contact_number:
            m = re.search(r"(0\d[\d\s\-]{8,12})", text)
            if m:
                self.slots.contact_number = m.group(1).strip()
            return

        if state == State.BOOKING_SERVICE and not self.slots.service_type:
            self.slots.service_type = text.strip()[:80]
            return

        if state == State.BOOKING_DATETIME:
            parsed = parse_date(text)
            now = datetime.now()
            if parsed:
                if parsed.date() < now.date():
                    return
                if not self.slots.preferred_date:
                    self.slots.preferred_date = parsed.strftime("%A, %d %B %Y")
            t = re.search(r"\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", lower)
            if t and not self.slots.preferred_time:
                self.slots.preferred_time = t.group(0)
            return

        if any(w in lower for w in ["book", "schedule", "appointment", "see a dentist",
                                     "checkup", "check-up", "check up", "cleaning",
                                     "braces", "whitening", "extraction", "filling",
                                     "consultation", "root canal", "need a", "i need", "i want"]):
            self.slots.intent = "book"
        elif any(w in lower for w in ["reschedule", "change my appointment"]):
            self.slots.intent = "reschedule"
        elif "cancel" in lower:
            self.slots.intent = "cancel"

        if not self.slots.patient_name:
            m = re.search(r"(?:my name is|i'm|i am|call me|this is)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})", text, re.IGNORECASE)
            if m:
                self.slots.patient_name = m.group(1).strip().title()

        m = re.search(r"(0\d[\d\s\-]{8,12})", text)
        if m and not self.slots.contact_number:
            self.slots.contact_number = m.group(1).strip()

        for svc in ["check-up", "checkup", "cleaning", "braces", "whitening", "extraction", "filling", "consultation", "root canal"]:
            if svc in lower and not self.slots.service_type:
                self.slots.service_type = svc

        parsed = parse_date(text)
        if parsed and not self.slots.preferred_date:
            if parsed.date() >= datetime.now().date():
                self.slots.preferred_date = parsed.strftime("%A, %d %B %Y")

        t = re.search(r"\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", lower)
        if t and not self.slots.preferred_time:
            self.slots.preferred_time = t.group(0)


# ══════════════════════════════════════════════════════════════════════════════
# Conversation Manager  (updated: session_id + tool dispatch)
# ══════════════════════════════════════════════════════════════════════════════

class ConversationManager:
    def __init__(
        self,
        llm_engine: LocalLLMEngine,
        session_id: str = "",
        orchestrator: Optional[ToolOrchestrator] = None,
    ) -> None:
        self.llm_engine = llm_engine
        self.session_id = session_id
        self.orchestrator = orchestrator
        self._init_session()

    def _init_session(self) -> None:
        self.state = State.GREETING
        self.turn_count = 0
        self.total_tokens = 0
        self.memory = ContextMemoryManager(
            token_counter=self.llm_engine.count_tokens,
            system_prompt=BASE_SYSTEM_PROMPT,
            max_turns=10,
            token_budget=1800,
        )

    def reset(self) -> None:
        self._init_session()

    def _build_prompt(self) -> str:
        state_suffix = STATE_PROMPTS.get(self.state.name, "")
        slot_context = self.memory.slots.to_context_string()

        parts = [BASE_SYSTEM_PROMPT]
        if state_suffix:
            parts.append(state_suffix)
        if slot_context:
            parts.append(f"\n{slot_context}")
        return "\n".join(parts)

    def _enforce_policies(self, response: str) -> str:
        lower = response.lower()
        diagnosis_phrases = [
            "you have a cavity",
            "you've got gum disease",
            "you might have an infection",
            "it's likely a cavity",
            "i think you have",
            "likely gum disease",
            "you have gum disease",
        ]
        for phrase in diagnosis_phrases:
            if phrase in lower:
                return (
                    "I'm not able to diagnose dental conditions - that requires "
                    "an in-person examination by one of our dentists. "
                    "Would you like me to book the earliest available slot?"
                )

        if any(k in lower for k in ["emergency", "severe pain", "urgent"]):
            if "042-35001234" not in response:
                response += " For immediate assistance please call: 042-35001234."
        return response.strip()

    def chat(self, user_input: str) -> tuple[str, State]:
        if self.state == State.END:
            return "(Session ended. Please start a new session.)", self.state

        intent = detect_intent(user_input)
        prev_state = self.state
        self.state = next_state(self.state, intent, self.memory.slots)

        self.memory.current_state = prev_state
        self.memory.add_user(user_input)

        messages = self.memory.get_messages()
        prompt = self._build_prompt()

        # ── RAG retrieval ───────────────────────────────────────────────────
        retrieved_chunks = retrieve_relevant_chunks(user_input, top_k=3)
        retrieved_text = "\n".join(retrieved_chunks) if retrieved_chunks else ""
        if retrieved_text:
            prompt += f"\n\n[KNOWLEDGE BASE CONTEXT]\n{retrieved_text}\n[END CONTEXT]"
            print(f"\n[RAG] Retrieved {len(retrieved_chunks)} chunks for: '{user_input[:50]}'")
        else:
            print(f"\n[RAG] No chunks for: '{user_input[:50]}'")

        # ── Tool dispatch ───────────────────────────────────────────────────
        if self.orchestrator:
            tool_context = self.orchestrator.dispatch(
                session_id=self.session_id,
                user_input=user_input,
                intent=intent,
                prev_state=prev_state,
                new_state=self.state,
                slots=self.memory.slots,
            )
            if tool_context:
                prompt += f"\n\n[TOOL CONTEXT — ground your response in this data]\n{tool_context}"
                print(f"[TOOLS] Context injected ({len(tool_context)} chars)")

        messages[0]["content"] = prompt

        if self.llm_engine.enabled:
            print("[RESPONSE] Using LLM Engine")
            result = self.llm_engine.generate(messages)
            reply = str(result.get("content", "")).strip()
            self.total_tokens += int(result.get("completion_tokens", 0))
        else:
            print("[RESPONSE] Using Fallback Engine (LLM not configured)")
            reply = self._fallback_reply(user_input, intent)

        print(f"[REPLY] {reply[:100]}...")
        reply = self._enforce_policies(reply)
        self.memory.add_assistant(reply)
        self.turn_count += 1
        return reply, self.state

    def _fallback_reply(self, user_input: str, intent: str) -> str:
        s = self.memory.slots
        lower = user_input.lower()

        if intent == "memory_query":
            return self._memory_recap()

        if intent == "chit_chat":
            if any(w in lower for w in ["thanks", "thank you"]):
                return "You're welcome. If you'd like, I can help you book or reschedule an appointment."
            if "how are you" in lower:
                return "I'm doing well, thanks. How can I help you with your dental appointment today?"
            return "Hi, glad you're here. Tell me what you need and I'll handle it step by step."

        if self.state == State.INTENT_DETECTION:
            if intent == "patient_status":
                status = "new" if s.patient_status == "new" else "existing"
                article = "a" if status == "new" else "an"
                return (
                    f"Thanks for confirming you're {article} {status} patient. "
                    "How can I help next: book, reschedule, cancel, or answer a clinic question?"
                )
            if "am i a new patient" in lower or "am i an existing patient" in lower:
                return (
                    "If this is your first visit, you're a new patient. If you've visited us before, you're an existing patient. "
                    "Would you like me to help you book an appointment?"
                )
            return "Welcome to BrightSmile Dental Clinic. Are you a new patient or an existing patient?"
        if self.state == State.BOOKING_NAME:
            if s.patient_name:
                return "Great, thanks. Could you share your contact number so I can continue?"
            return "Perfect. Could I have your full name?"
        if self.state == State.BOOKING_CONTACT:
            if s.contact_number:
                return "Perfect. What type of appointment do you need?"
            return "Could you share your contact number?"
        if self.state == State.BOOKING_SERVICE:
            if s.service_type:
                return "Got it. Please choose a future date and time that works for you."
            return "What type of appointment do you need? For example, check-up, cleaning, braces, or consultation."
        if self.state == State.BOOKING_DATETIME:
            if s.preferred_date and s.preferred_time:
                return (
                    f"Please confirm this booking: {s.patient_name or 'N/A'}, {s.service_type or 'N/A'}, "
                    f"{s.preferred_date} at {s.preferred_time}."
                )
            return "Please choose a future date and time. I can offer Saturday 10:00 AM, Monday 2:30 PM, or Tuesday 11:00 AM."
        if self.state == State.BOOKING_CONFIRM:
            return (
                f"Please confirm your booking: {s.patient_name or 'N/A'}, {s.service_type or 'N/A'}, "
                f"{s.preferred_date or 'N/A'} at {s.preferred_time or 'N/A'}."
            )
        if self.state == State.BOOKING_DONE:
            return "Your appointment is confirmed. You will receive a reminder. Can I help with anything else?"
        if self.state == State.RESCHEDULE:
            return "I can help reschedule. Please share your name and current appointment date."
        if self.state == State.CANCEL:
            return "I can help cancel your appointment. Please share your name and appointment date."
        if self.state == State.FAQ:
            lower = user_input.lower()
            if "new patient" in lower or "existing patient" in lower:
                return (
                    "If this is your first visit, you're a new patient. "
                    "If you've already visited us before, you're an existing patient. "
                    "Would you like me to book your appointment now?"
                )
            if "hours" in lower:
                return "We're open Monday to Saturday, 9:00 AM to 7:00 PM. Want me to help you book a slot?"
            if "insurance" in lower:
                return "We accept Jubilee Health, EFU, Adamjee, and State Life. Want me to schedule an appointment for you?"
            if "location" in lower or "where" in lower:
                return "We're located at 42 Gulberg III, Lahore. Would you like directions or a booking?"
            return "I can help with services, clinic hours, insurance, and location. What would you like to know?"
        if self.state == State.EMERGENCY:
            return "I am sorry you are dealing with this. I can arrange the earliest slot. For immediate assistance call 042-35001234."
        if self.state == State.CLOSING:
            return "Thank you for contacting BrightSmile Dental Clinic. If you need anything, call 042-35001234."
        return "I can help with booking, rescheduling, cancellations, and clinic questions. What do you want to do?"

    def _memory_recap(self) -> str:
        details: List[str] = []
        s = self.memory.slots
        if s.patient_status:
            article = "a" if s.patient_status == "new" else "an"
            details.append(f"you're {article} {s.patient_status} patient")
        if s.patient_name:
            details.append(f"your name is {s.patient_name}")
        if s.contact_number:
            details.append(f"your contact is {s.contact_number}")
        if s.service_type:
            details.append(f"you need {s.service_type}")
        if s.preferred_date or s.preferred_time:
            when = " ".join([p for p in [s.preferred_date, s.preferred_time] if p])
            details.append(f"you prefer {when}")

        if details:
            return "From this chat, I have: " + "; ".join(details) + ". Want me to continue from here?"

        recent_user_msgs = [m["content"] for m in self.memory.history if m["role"] == "user"]
        recent_user_msgs = recent_user_msgs[-3:-1] if len(recent_user_msgs) > 1 else recent_user_msgs[-2:]
        if recent_user_msgs:
            joined = " | ".join(recent_user_msgs)
            return f"You said: {joined}. I can also keep track of booking details as we continue."

        return "We just started, so I don't have much context yet. Tell me what you'd like help with."


# ══════════════════════════════════════════════════════════════════════════════
# Session Store (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationSession:
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    history: List[Dict[str, str]] = field(default_factory=list)
    state: SessionState = SessionState.greeting
    turn_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset(self) -> None:
        self.history.clear()
        self.state = SessionState.greeting
        self.turn_count = 0
        self.updated_at = datetime.now(timezone.utc)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> ConversationSession:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id=session_id)
            return self._sessions[session_id]


# ══════════════════════════════════════════════════════════════════════════════
# DentaBot Engine  (updated: creates ToolOrchestrator, passes to managers)
# ══════════════════════════════════════════════════════════════════════════════

class DentaBotEngine:
    """Phase IV async engine — RAG + Tools + optional local GGUF LLM."""

    def __init__(self) -> None:
        self.llm = LocalLLMEngine()
        self.orchestrator = ToolOrchestrator()
        self._managers: Dict[str, ConversationManager] = {}

    async def chat(self, session: ConversationSession, user_message: str) -> str:
        manager = self._managers.get(session.session_id)
        if manager is None:
            manager = ConversationManager(
                self.llm,
                session_id=session.session_id,
                orchestrator=self.orchestrator,
            )
            self._managers[session.session_id] = manager

        session.updated_at = datetime.now(timezone.utc)
        session.turn_count += 1
        session.history.append({"role": "user", "content": user_message})

        reply, phase_state = await asyncio.to_thread(manager.chat, user_message)
        session.state = _api_state(phase_state)

        session.history.append({"role": "assistant", "content": reply})
        return reply

    async def stream_tokens(self, reply: str) -> AsyncGenerator[str, None]:
        for token in re.findall(r"\S+\s*", reply):
            yield token
            await asyncio.sleep(0)

    def reset_session(self, session_id: str) -> None:
        manager = self._managers.get(session_id)
        if manager is not None:
            manager.reset()

    @property
    def backend(self) -> str:
        return "llama-cpp" if self.llm.enabled else "rule-fallback"