import json
import os
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

SCHEMA_VERSION = "1.0"
FALLBACK_MOOD = "balanced"
CONFIDENCE_FALLBACK_THRESHOLD = 0.55
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

ALLOWED_MOODS = {
    "happy",
    "chill",
    "relaxed",
    "moody",
    "sad",
    "intense",
    "focused",
    "nostalgic",
    "balanced",
}

# Weighted keyword map used by the local mood parser.
MOOD_KEYWORDS: Dict[str, Dict[str, float]] = {
    "happy": {
        "happy": 1.2,
        "joy": 1.0,
        "joyful": 1.1,
        "upbeat": 1.1,
        "sunny": 0.9,
        "cheerful": 1.0,
        "fun": 0.7,
        "party": 0.8,
        "euphoric": 1.1,
    },
    "chill": {
        "chill": 1.2,
        "laidback": 1.1,
        "easy": 0.7,
        "calm": 0.8,
        "breezy": 0.8,
        "lofi": 0.9,
        "study": 0.8,
    },
    "relaxed": {
        "relaxed": 1.2,
        "relax": 1.0,
        "soothing": 1.0,
        "unwind": 1.0,
        "gentle": 0.8,
        "mellow": 0.9,
    },
    "moody": {
        "moody": 1.3,
        "brooding": 1.2,
        "dark": 0.9,
        "atmospheric": 0.9,
        "vibe": 0.6,
        "night": 0.6,
    },
    "sad": {
        "sad": 1.3,
        "heartbreak": 1.2,
        "cry": 1.0,
        "lonely": 1.0,
        "melancholy": 1.2,
        "blue": 0.8,
    },
    "intense": {
        "intense": 1.3,
        "hype": 1.2,
        "power": 1.0,
        "hard": 0.9,
        "aggressive": 1.1,
        "workout": 1.2,
        "pump": 1.1,
    },
    "focused": {
        "focused": 1.3,
        "focus": 1.2,
        "concentration": 1.1,
        "deep": 0.8,
        "coding": 0.8,
        "productive": 1.1,
        "instrumental": 0.8,
    },
    "nostalgic": {
        "nostalgic": 1.3,
        "nostalgia": 1.3,
        "throwback": 1.1,
        "retro": 1.1,
        "old": 0.6,
        "memories": 1.0,
        "classic": 0.8,
    },
}

HIGH_ENERGY_HINTS = {"hype", "workout", "party", "dance", "running", "sprint", "intense", "pump"}
LOW_ENERGY_HINTS = {"sleep", "study", "focus", "calm", "soft", "quiet", "relax", "chill", "lofi", "unwind"}

ENERGY_HINT_BY_MOOD = {
    "happy": 0.72,
    "chill": 0.36,
    "relaxed": 0.32,
    "moody": 0.48,
    "sad": 0.34,
    "intense": 0.88,
    "focused": 0.45,
    "nostalgic": 0.50,
    "balanced": 0.55,
}


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _score_moods(tokens: List[str]) -> Dict[str, float]:
    scores = {mood: 0.0 for mood in ALLOWED_MOODS if mood != FALLBACK_MOOD}
    for token in tokens:
        for mood, weighted_keywords in MOOD_KEYWORDS.items():
            scores[mood] += weighted_keywords.get(token, 0.0)
    return scores


def _estimate_confidence(top_score: float, second_score: float) -> float:
    if top_score <= 0.0:
        return 0.0
    margin = max(0.0, top_score - second_score)
    confidence = 0.40 + min(top_score * 0.14, 0.40) + min(margin * 0.25, 0.20)
    return _clamp_01(confidence)


def _choose_energy_hint(tokens: List[str], detected_mood: str, confidence: float) -> Optional[float]:
    if any(token in HIGH_ENERGY_HINTS for token in tokens):
        return 0.85
    if any(token in LOW_ENERGY_HINTS for token in tokens):
        return 0.30
    if confidence >= CONFIDENCE_FALLBACK_THRESHOLD:
        return ENERGY_HINT_BY_MOOD.get(detected_mood)
    return None


def _build_notes(
    detected_mood: str,
    confidence: float,
    top_mood: str,
    top_score: float,
    tokens: List[str],
) -> str:
    if detected_mood == FALLBACK_MOOD:
        return "low-confidence fallback to balanced"
    if top_score <= 0.0 or not tokens:
        return "no strong mood keywords detected"
    return f"keyword-based mood match: {top_mood}"


def _coerce_allowed_moods(raw_candidates: Any) -> List[str]:
    if not isinstance(raw_candidates, list):
        return []

    normalized: List[str] = []
    for candidate in raw_candidates:
        mood = str(candidate).strip().lower()
        if mood in ALLOWED_MOODS and mood not in normalized:
            normalized.append(mood)
    return normalized


def _extract_json_content(raw_content: str) -> Optional[Dict[str, Any]]:
    content = (raw_content or "").strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if block_match:
        try:
            parsed = json.loads(block_match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    object_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if object_match:
        try:
            parsed = json.loads(object_match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def _gemini_prompt(user_message: str, optional_context: Optional[Dict[str, Any]]) -> str:
    context = optional_context or {}
    return (
        "You are Agent 1 mood parser for a music recommender. "
        "Return strict JSON only (no markdown) with keys: "
        "detected_mood, confidence, energy_hint, mood_candidates, notes. "
        "Rules: detected_mood must be one of "
        "[happy,chill,relaxed,moody,sad,intense,focused,nostalgic,balanced]. "
        "confidence must be a float between 0 and 1. energy_hint must be null or float between 0 and 1. "
        "mood_candidates must be up to 3 moods from the allowed set. "
        f"User message: {user_message!r}. "
        f"Optional context: {context!r}."
    )


def _local_analyze_mood(user_message: str, optional_context: Optional[Dict[str, Any]], trace_id: Optional[str]) -> Dict[str, Any]:
    text = (user_message or "").strip()
    context = optional_context or {}
    tokens = _tokenize(text)

    scores = _score_moods(tokens)

    # Optional context can lightly bias mood ranking when available.
    prior_mood = str(context.get("prior_mood", "")).lower()
    if prior_mood in scores:
        scores[prior_mood] += 0.25

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_mood, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = _estimate_confidence(top_score, second_score)
    detected_mood = top_mood if confidence >= CONFIDENCE_FALLBACK_THRESHOLD else FALLBACK_MOOD

    candidates = [mood for mood, score in ranked if score > 0][:3]
    if not candidates:
        candidates = [FALLBACK_MOOD]

    energy_hint = _choose_energy_hint(tokens, detected_mood, confidence)
    notes = _build_notes(detected_mood, confidence, top_mood, top_score, tokens)

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid4()),
        "detected_mood": detected_mood,
        "confidence": round(_clamp_01(confidence), 4),
        "energy_hint": energy_hint,
        "mood_candidates": candidates,
        "notes": notes,
    }


def _gemini_analyze_mood(
    user_message: str,
    optional_context: Optional[Dict[str, Any]],
    trace_id: Optional[str],
    model: str,
    api_key: Optional[str],
    llm_class: Any,
    message_class: Any,
) -> Dict[str, Any]:
    llm = llm_class(model=model, google_api_key=api_key, temperature=0)
    response = llm.invoke([message_class(content=_gemini_prompt(user_message, optional_context))])
    response_content = str(getattr(response, "content", response))

    payload = _extract_json_content(response_content)
    if payload is None:
        raise ValueError("Gemini response did not contain valid JSON object")

    raw_mood = str(payload.get("detected_mood", FALLBACK_MOOD)).lower()
    raw_confidence = payload.get("confidence", 0.0)
    raw_energy_hint = payload.get("energy_hint")
    raw_candidates = payload.get("mood_candidates", [])

    detected_mood = raw_mood if raw_mood in ALLOWED_MOODS else FALLBACK_MOOD

    try:
        confidence = _clamp_01(float(raw_confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence < CONFIDENCE_FALLBACK_THRESHOLD:
        detected_mood = FALLBACK_MOOD

    try:
        energy_hint = _clamp_01(float(raw_energy_hint)) if raw_energy_hint is not None else None
    except (TypeError, ValueError):
        energy_hint = None

    candidates = _coerce_allowed_moods(raw_candidates)[:3]
    if not candidates:
        candidates = [detected_mood]

    notes = str(payload.get("notes", "gemini mood parse")).strip() or "gemini mood parse"

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid4()),
        "detected_mood": detected_mood,
        "confidence": round(confidence, 4),
        "energy_hint": energy_hint,
        "mood_candidates": candidates,
        "notes": notes,
    }


def analyze_mood(
    user_message: str,
    optional_context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    backend: str = "local",
    model: str = DEFAULT_GEMINI_MODEL,
    api_key: Optional[str] = None,
    llm_class: Any = None,
    message_class: Any = None,
) -> Dict[str, Any]:
    selected_backend = (backend or "local").strip().lower()

    if selected_backend == "local":
        return _local_analyze_mood(user_message, optional_context, trace_id)

    if selected_backend not in {"gemini", "auto"}:
        raise ValueError("backend must be one of: local, gemini, auto")

    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    should_try_gemini = selected_backend == "gemini" or bool(resolved_key)

    if should_try_gemini:
        try:
            if llm_class is None or message_class is None:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage

                llm_class = ChatGoogleGenerativeAI
                message_class = HumanMessage

            if not resolved_key:
                raise ValueError("missing GOOGLE_API_KEY")

            return _gemini_analyze_mood(
                user_message=user_message,
                optional_context=optional_context,
                trace_id=trace_id,
                model=model,
                api_key=resolved_key,
                llm_class=llm_class,
                message_class=message_class,
            )
        except Exception as exc:
            local_payload = _local_analyze_mood(user_message, optional_context, trace_id)
            local_payload["notes"] = f"{local_payload['notes']} | gemini fallback: {exc}"
            return local_payload

    return _local_analyze_mood(user_message, optional_context, trace_id)


class MoodAnalyst:
    """Agent 1 that converts free text into a normalized mood payload."""

    def analyze(
        self,
        user_message: str,
        optional_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        backend: str = "local",
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return analyze_mood(
            user_message=user_message,
            optional_context=optional_context,
            trace_id=trace_id,
            backend=backend,
            model=model,
            api_key=api_key,
        )
