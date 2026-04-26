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
        "excited": 1.0,
        "great": 0.8,
        "amazing": 0.9,
        "good": 0.6,
        "blessed": 0.9,
        "vibing": 0.8,
        "lit": 0.7,
        "celebrate": 1.0,
    },
    "chill": {
        "chill": 1.2,
        "laidback": 1.1,
        "easy": 0.7,
        "calm": 0.8,
        "breezy": 0.8,
        "lofi": 0.9,
        "study": 0.8,
        "cozy": 0.9,
        "lazy": 0.8,
        "slow": 0.7,
        "sunday": 0.7,
        "coffee": 0.6,
        "afternoon": 0.6,
    },
    "relaxed": {
        "relaxed": 1.2,
        "relax": 1.0,
        "soothing": 1.0,
        "unwind": 1.0,
        "gentle": 0.8,
        "mellow": 0.9,
        "peaceful": 1.0,
        "wind": 0.7,
        "breathe": 0.8,
        "rest": 0.8,
        "calmdown": 0.9,
    },
    "moody": {
        "moody": 1.3,
        "brooding": 1.2,
        "dark": 0.9,
        "atmospheric": 0.9,
        "vibe": 0.6,
        "night": 0.6,
        "cloudy": 0.7,
        "complicated": 0.8,
        "complex": 0.7,
        "introspective": 1.0,
        "overthinking": 0.9,
    },
    "sad": {
        "sad": 1.3,
        "heartbreak": 1.2,
        "cry": 1.0,
        "lonely": 1.0,
        "melancholy": 1.2,
        "blue": 0.8,
        "down": 0.8,
        "depressed": 1.1,
        "hurt": 0.9,
        "miss": 0.8,
        "lost": 0.7,
        "broken": 1.0,
        "upset": 0.9,
        "crying": 1.1,
        "tears": 1.0,
    },
    "intense": {
        "intense": 1.3,
        "hype": 1.2,
        "power": 1.0,
        "hard": 0.9,
        "aggressive": 1.1,
        "workout": 1.2,
        "pump": 1.1,
        "gym": 1.3,
        "grind": 1.0,
        "beast": 1.1,
        "energy": 0.9,
        "fire": 1.0,
        "crush": 0.9,
        "push": 0.9,
        "training": 1.1,
        "lifting": 1.1,
        "run": 0.9,
        "running": 1.0,
        "sprint": 1.1,
        "motivated": 1.0,
        "lets": 0.7,
        "go": 0.6,
        "hit": 0.8,
    },
    "focused": {
        "focused": 1.3,
        "focus": 1.2,
        "concentration": 1.1,
        "deep": 0.8,
        "coding": 0.8,
        "productive": 1.1,
        "instrumental": 0.8,
        "lock": 1.2,
        "locked": 1.2,
        "grind": 0.9,
        "hustle": 1.0,
        "work": 0.9,
        "working": 0.9,
        "task": 0.9,
        "deadline": 1.1,
        "studying": 1.0,
        "homework": 0.9,
        "concentrate": 1.1,
        "dialed": 1.0,
        "mode": 0.7,
        "flow": 0.8,
        "zone": 0.9,
    },
    "nostalgic": {
        "nostalgic": 1.3,
        "nostalgia": 1.3,
        "throwback": 1.1,
        "retro": 1.1,
        "old": 0.6,
        "memories": 1.0,
        "classic": 0.8,
        "remember": 0.9,
        "childhood": 1.0,
        "back": 0.6,
        "miss": 0.7,
        "used": 0.5,
        "days": 0.5,
    },
}

# Multi-word phrase → mood boost applied before token scoring.
PHRASE_MOOD_MAP: Dict[str, tuple] = {
    "hit the gym": ("intense", 2.0),
    "hitting the gym": ("intense", 2.0),
    "at the gym": ("intense", 1.5),
    "going to the gym": ("intense", 2.0),
    "lock in": ("focused", 2.0),
    "locking in": ("focused", 2.0),
    "locked in": ("focused", 2.0),
    "get stuff done": ("focused", 1.8),
    "get things done": ("focused", 1.8),
    "grind time": ("focused", 1.8),
    "in my feels": ("moody", 1.8),
    "feeling myself": ("happy", 1.5),
    "good vibes": ("happy", 1.5),
    "feeling down": ("sad", 1.8),
    "feeling sad": ("sad", 2.0),
    "need to cry": ("sad", 2.0),
    "wind down": ("relaxed", 1.8),
    "take it easy": ("relaxed", 1.5),
    "chill out": ("chill", 1.8),
    "kick back": ("chill", 1.5),
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


def _score_moods(tokens: List[str], text: str = "") -> Dict[str, float]:
    scores = {mood: 0.0 for mood in ALLOWED_MOODS if mood != FALLBACK_MOOD}
    for token in tokens:
        for mood, weighted_keywords in MOOD_KEYWORDS.items():
            scores[mood] += weighted_keywords.get(token, 0.0)
    normalized = text.lower()
    for phrase, (mood, boost) in PHRASE_MOOD_MAP.items():
        if phrase in normalized:
            scores[mood] += boost
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
        "You are Agent 1, a mood and audio-profile parser for a music recommender. "
        "Your job is to infer the user's full music preference profile from casual, everyday language. "
        "Interpret colloquial and slang phrases semantically — for example: "
        "'hitting the gym' or 'beast mode' → intense, high energy, high danceability, fast tempo; "
        "'gotta lock in' or 'need to focus' → focused, low-medium energy, high instrumentalness; "
        "'in my feels' or 'going through it' → moody, low energy, low valence; "
        "'good vibes only' or 'feeling myself' → happy, medium-high energy, high valence; "
        "'wind down' or 'take it easy' → relaxed, low energy, high acousticness. "
        "Return strict JSON only (no markdown) with ALL of the following keys:\n"
        "  detected_mood: one of [happy,chill,relaxed,moody,sad,intense,focused,nostalgic,balanced]\n"
        "  confidence: float [0,1]\n"
        "  energy_hint: float [0,1] (0=very calm, 1=maximum energy)\n"
        "  mood_candidates: list of up to 3 moods from the allowed set\n"
        "  notes: brief plain-English reasoning\n"
        "  target_energy: float [0,1]\n"
        "  target_valence: float [0,1] (0=dark/negative, 1=bright/positive)\n"
        "  target_danceability: float [0,1]\n"
        "  target_tempo_bpm: float (beats per minute, typical range 60-180)\n"
        "  target_acousticness: float [0,1] (0=electronic/produced, 1=acoustic/raw)\n"
        "  target_instrumentalness: float [0,1] (0=vocal-heavy, 1=fully instrumental)\n"
        "  target_brightness: float [0,1] (0=dark/muffled, 1=bright/crisp)\n"
        "  favorite_genre: one of [pop,rock,indie,indie pop,hip-hop,r&b,edm,lofi,jazz,classical,country,metal,ambient,synthpop] or null\n"
        "  likes_acoustic: bool\n"
        "  avoid_genres: list of genre strings the user wants excluded (empty list if none)\n"
        f"User message: {user_message!r}. "
        f"Optional context: {context!r}."
    )


def _local_analyze_mood(user_message: str, optional_context: Optional[Dict[str, Any]], trace_id: Optional[str]) -> Dict[str, Any]:
    text = (user_message or "").strip()
    context = optional_context or {}
    tokens = _tokenize(text)

    scores = _score_moods(tokens, text)

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
    llm = llm_class(model=model, google_api_key=api_key, temperature=0, max_output_tokens=256)
    response = llm.invoke([message_class(content=_gemini_prompt(user_message, optional_context))])
    raw = getattr(response, "content", response)
    if isinstance(raw, list):
        response_content = " ".join(b.get("text", "") for b in raw if isinstance(b, dict) and "text" in b)
    else:
        response_content = str(raw)

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

    def _safe_clamp(key: str, default: Optional[float]) -> Optional[float]:
        val = payload.get(key)
        if val is None:
            return default
        try:
            return _clamp_01(float(val))
        except (TypeError, ValueError):
            return default

    def _safe_float_field(key: str, default: Optional[float]) -> Optional[float]:
        val = payload.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    llm_genre = payload.get("favorite_genre")
    llm_avoid = payload.get("avoid_genres", [])

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id or str(uuid4()),
        "detected_mood": detected_mood,
        "confidence": round(confidence, 4),
        "energy_hint": energy_hint,
        "mood_candidates": candidates,
        "notes": notes,
        # LLM-assigned audio feature targets — consumed by Agent 2 in llm mode
        "target_energy": _safe_clamp("target_energy", None),
        "target_valence": _safe_clamp("target_valence", None),
        "target_danceability": _safe_clamp("target_danceability", None),
        "target_tempo_bpm": _safe_float_field("target_tempo_bpm", None),
        "target_acousticness": _safe_clamp("target_acousticness", None),
        "target_instrumentalness": _safe_clamp("target_instrumentalness", None),
        "target_brightness": _safe_clamp("target_brightness", None),
        "favorite_genre": llm_genre if isinstance(llm_genre, str) and llm_genre else None,
        "likes_acoustic": bool(payload.get("likes_acoustic", False)),
        "avoid_genres": [g for g in llm_avoid if isinstance(g, str)] if isinstance(llm_avoid, list) else [],
        "llm_profile": True,
    }


_ST_MODEL_NAME = "all-MiniLM-L6-v2"
_st_model_cache = None

_ST_MOOD_REFERENCES: Dict[str, List[str]] = {
    "happy": [
        "I feel happy joyful euphoric and excited, ready to celebrate and have fun with friends",
        "upbeat cheerful sunny blessed vibing lit amazing great party celebrate joyful",
        "I am in a great mood, feeling positive, bright and energetic",
    ],
    "chill": [
        "I want to chill out with lofi beats, laid-back mellow background music while relaxing",
        "chill lofi mellow laidback breezy casual easy vibes coffee afternoon Sunday",
        "just vibing, something chill and relaxed, nothing too intense, lofi hip hop",
    ],
    "relaxed": [
        "I need to unwind and decompress after a long day, something peaceful and soothing",
        "unwind calm peaceful soothing gentle quiet tranquil wind down rest spa ambient",
        "slow calm music to help me rest, de-stress and feel at ease, very gentle",
    ],
    "moody": [
        "I am feeling dark brooding atmospheric and introspective, late night deep thoughts",
        "moody dark atmospheric cloudy brooding melancholic night drive indie alternative",
        "complex emotional music, bittersweet, contemplative, rainy window, introspective",
    ],
    "sad": [
        "I feel sad heartbroken lonely and want to cry, emotional grief and melancholy",
        "sad heartbreak lonely cry tears emotional melancholy loss grief blue depressed",
        "I am going through something hard, feeling down, music that matches my sadness",
    ],
    "intense": [
        "I need hype intense aggressive high energy music for a workout sprint or game",
        "intense hype pump workout gym beast mode aggressive sprint training power adrenaline",
        "fast loud powerful energetic music, pushing limits, going hard, maximum energy",
    ],
    "focused": [
        "I need to concentrate and focus on studying coding or deep work with no distractions",
        "focus concentrate study coding work deadline productive deep work flow state instrumental",
        "background music to help me think clearly and stay in the zone while working",
    ],
    "nostalgic": [
        "I want throwback retro classic memories from the old days, childhood nostalgia",
        "nostalgic throwback retro classic old school memories childhood vintage 90s 80s",
        "songs that remind me of the past, good old times, sentimental memories",
    ],
    "balanced": [
        "just play me something nice, anything good, no strong preference",
        "neutral balanced mixed general background pleasant surprise me anything",
    ],
}

_ST_CONFIDENCE_THRESHOLD = 0.38  # cosine similarity range differs from Gemini confidence


def _get_st_model():
    global _st_model_cache
    if _st_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _st_model_cache = SentenceTransformer(_ST_MODEL_NAME)
    return _st_model_cache


def _st_analyze_mood(
    user_message: str,
    optional_context: Optional[Dict[str, Any]],
    trace_id: Optional[str],
) -> Dict[str, Any]:
    import numpy as np
    model = _get_st_model()

    moods = list(_ST_MOOD_REFERENCES.keys())

    # Encode all reference sentences in one batch, then average per mood.
    all_refs = [s for refs in _ST_MOOD_REFERENCES.values() for s in refs]
    ref_counts = [len(refs) for refs in _ST_MOOD_REFERENCES.values()]

    all_sentences = [user_message] + all_refs
    embeddings = model.encode(all_sentences, normalize_embeddings=True, show_progress_bar=False)

    user_emb = embeddings[0]
    ref_embs = embeddings[1:]

    # Average embeddings per mood, then re-normalize.
    mood_scores = []
    offset = 0
    for count in ref_counts:
        mood_emb = ref_embs[offset:offset + count].mean(axis=0)
        norm = np.linalg.norm(mood_emb)
        if norm > 0:
            mood_emb = mood_emb / norm
        mood_scores.append(float(user_emb @ mood_emb))
        offset += count

    best_idx = int(max(range(len(mood_scores)), key=lambda i: mood_scores[i]))
    best_mood = moods[best_idx]
    best_score = mood_scores[best_idx]

    sorted_pairs = sorted(zip(moods, mood_scores), key=lambda x: x[1], reverse=True)
    candidates = [m for m, _ in sorted_pairs[:3]]

    confidence = round(float(best_score), 4)
    detected_mood = best_mood if confidence >= _ST_CONFIDENCE_THRESHOLD else FALLBACK_MOOD

    energy_hint = {
        "happy": 0.75, "intense": 0.90, "chill": 0.40, "relaxed": 0.30,
        "moody": 0.50, "sad": 0.35, "focused": 0.45, "nostalgic": 0.55,
        "balanced": 0.55,
    }.get(detected_mood, 0.55)

    resolved_trace = trace_id or str(uuid4())
    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "detected_mood": detected_mood,
        "confidence": confidence,
        "energy_hint": energy_hint,
        "mood_candidates": candidates,
        "notes": f"sentence-transformer:{_ST_MODEL_NAME} score={confidence:.3f}",
        "target_energy": None,
        "target_valence": None,
        "target_danceability": None,
        "target_tempo_bpm": None,
        "target_acousticness": None,
        "target_instrumentalness": None,
        "target_brightness": None,
        "favorite_genre": None,
        "likes_acoustic": False,
        "avoid_genres": [],
        "llm_profile": False,
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

    if selected_backend == "sentence_transformers":
        return _st_analyze_mood(user_message, optional_context, trace_id)

    if selected_backend not in {"gemini", "auto"}:
        raise ValueError("backend must be one of: local, gemini, auto, sentence_transformers")

    resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
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
