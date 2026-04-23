import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .agent1_mood import ALLOWED_MOODS, CONFIDENCE_FALLBACK_THRESHOLD

SCHEMA_VERSION = "1.0"
DEFAULT_PROFILE = {
    "favorite_genre": "pop",
    "favorite_mood": "balanced",
    "target_energy": 0.55,
    "likes_acoustic": False,
    "avoid_genres": [],
}

GENRE_ALIASES = {
    "hip hop": "hip-hop",
    "hiphop": "hip-hop",
    "rap": "hip-hop",
    "rnb": "r&b",
    "rhythm and blues": "r&b",
    "electronic dance": "edm",
}

KNOWN_GENRES = {
    "pop",
    "rock",
    "indie",
    "indie pop",
    "hip-hop",
    "r&b",
    "edm",
    "electronic",
    "lofi",
    "jazz",
    "classical",
    "country",
    "metal",
    "ambient",
    "synthpop",
}

MOOD_ALIASES = {
    "upbeat": "happy",
    "cheerful": "happy",
    "calm": "relaxed",
    "mellow": "relaxed",
    "dark": "moody",
    "hype": "intense",
}

MOOD_KEYWORDS = {
    "happy": {"happy", "upbeat", "joyful", "cheerful", "fun"},
    "chill": {"chill", "laidback", "easy"},
    "relaxed": {"relaxed", "calm", "mellow", "gentle", "soothing", "unwind"},
    "moody": {"moody", "dark", "brooding", "atmospheric"},
    "sad": {"sad", "melancholy", "heartbreak", "cry", "lonely"},
    "intense": {"intense", "hype", "power", "aggressive", "workout", "pump"},
    "focused": {"focused", "focus", "study", "coding", "productive", "concentration"},
    "nostalgic": {"nostalgic", "retro", "throwback", "classic", "memories"},
}

HIGH_ENERGY_TOKENS = {"high", "hype", "intense", "workout", "run", "running", "party", "dance"}
LOW_ENERGY_TOKENS = {"low", "calm", "chill", "sleep", "study", "focus", "soft", "quiet", "relax"}

ACOUSTIC_TOKENS = {"acoustic", "unplugged", "stripped", "lofi"}
NON_ACOUSTIC_TOKENS = {"electronic", "edm", "club", "remix", "heavy bass", "synth"}

NEGATION_PREFIXES = ("no", "not", "avoid", "without", "skip")


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _normalize_genre(raw_genre: str) -> Optional[str]:
    clean = re.sub(r"\s+", " ", raw_genre.strip().lower())
    if not clean:
        return None
    if clean in GENRE_ALIASES:
        clean = GENRE_ALIASES[clean]
    return clean if clean in KNOWN_GENRES else None


def _extract_primary_genre(message: str) -> Optional[str]:
    lower = message.lower()
    # Multi-word genres first to prevent partial token collisions.
    ordered_candidates = sorted(KNOWN_GENRES.union(GENRE_ALIASES.keys()), key=len, reverse=True)
    for candidate in ordered_candidates:
        pattern = rf"\b{re.escape(candidate)}\b"
        if re.search(pattern, lower):
            return _normalize_genre(candidate)
    return None


def _extract_avoid_genres(message: str) -> List[str]:
    lower = message.lower()
    avoid_matches: List[tuple[int, str]] = []

    ordered_candidates = sorted(KNOWN_GENRES.union(GENRE_ALIASES.keys()), key=len, reverse=True)
    for candidate in ordered_candidates:
        normalized = _normalize_genre(candidate)
        if not normalized:
            continue

        for prefix in NEGATION_PREFIXES:
            pattern = rf"\b{prefix}\s+{re.escape(candidate)}\b"
            match = re.search(pattern, lower)
            if match:
                avoid_matches.append((match.start(), normalized))
                break

    avoid_matches.sort(key=lambda item: item[0])

    ordered_unique: List[str] = []
    for _, genre in avoid_matches:
        if genre not in ordered_unique:
            ordered_unique.append(genre)

    return ordered_unique


def _extract_explicit_mood(tokens: List[str]) -> Optional[str]:
    for token in tokens:
        if token in ALLOWED_MOODS:
            return token
        if token in MOOD_ALIASES:
            return MOOD_ALIASES[token]

    for mood, keywords in MOOD_KEYWORDS.items():
        if any(token in keywords for token in tokens):
            return mood

    return None


def _parse_explicit_energy(message: str, tokens: List[str]) -> Optional[float]:
    numeric_match = re.search(r"(?:energy\s*[:=]?\s*|)(0(?:\.\d+)?|1(?:\.0+)?)", message.lower())
    if numeric_match:
        return _clamp_01(float(numeric_match.group(1)))

    if any(token in HIGH_ENERGY_TOKENS for token in tokens):
        return 0.8
    if any(token in LOW_ENERGY_TOKENS for token in tokens):
        return 0.3
    if "medium" in tokens or "balanced" in tokens:
        return 0.55

    return None


def _infer_likes_acoustic(tokens: List[str], resolved_mood: str) -> bool:
    has_acoustic = any(token in ACOUSTIC_TOKENS for token in tokens)
    has_non_acoustic = any(token in NON_ACOUSTIC_TOKENS for token in tokens)

    if has_acoustic and not has_non_acoustic:
        return True
    if has_non_acoustic and not has_acoustic:
        return False

    if resolved_mood in {"chill", "relaxed", "sad", "focused", "nostalgic", "moody"}:
        return True

    return False


def parse_profile(
    user_message: str,
    agent1_payload: Dict[str, Any],
    optional_context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    message = (user_message or "").strip()
    tokens = _tokenize(message)
    context = optional_context or {}

    constraints: Dict[str, Any] = {
        "missing_fields": [],
        "inferred_fields": [],
        "low_confidence_mood": False,
        "disallowed_or_unknown_terms": [],
        "parser_mode": "rules",
    }

    incoming_trace = trace_id or agent1_payload.get("trace_id")
    resolved_trace = incoming_trace if isinstance(incoming_trace, str) and incoming_trace.strip() else str(uuid4())

    agent_mood = str(agent1_payload.get("detected_mood", "balanced")).lower()
    if agent_mood not in ALLOWED_MOODS:
        agent_mood = "balanced"

    raw_confidence = agent1_payload.get("confidence", 0.0)
    try:
        confidence = _clamp_01(float(raw_confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    explicit_mood = _extract_explicit_mood(tokens)
    if explicit_mood:
        favorite_mood = explicit_mood
    elif confidence >= CONFIDENCE_FALLBACK_THRESHOLD:
        favorite_mood = agent_mood
        constraints["inferred_fields"].append("favorite_mood")
    else:
        favorite_mood = "balanced"
        constraints["low_confidence_mood"] = True
        constraints["inferred_fields"].append("favorite_mood")

    genre = _extract_primary_genre(message)
    if genre is None:
        context_genre = _normalize_genre(str(context.get("favorite_genre", "")))
        if context_genre:
            genre = context_genre
            constraints["inferred_fields"].append("favorite_genre")

    if genre is None:
        genre = DEFAULT_PROFILE["favorite_genre"]
        constraints["missing_fields"].append("favorite_genre")

    explicit_energy = _parse_explicit_energy(message, tokens)
    if explicit_energy is not None:
        target_energy = explicit_energy
    else:
        hint = agent1_payload.get("energy_hint")
        try:
            target_energy = _clamp_01(float(hint)) if hint is not None else DEFAULT_PROFILE["target_energy"]
        except (TypeError, ValueError):
            target_energy = DEFAULT_PROFILE["target_energy"]

        constraints["inferred_fields"].append("target_energy")
        if hint is None:
            constraints["missing_fields"].append("target_energy")

    avoid_genres = _extract_avoid_genres(message)

    likes_acoustic = _infer_likes_acoustic(tokens, favorite_mood)
    if not any(token in ACOUSTIC_TOKENS or token in NON_ACOUSTIC_TOKENS for token in tokens):
        constraints["inferred_fields"].append("likes_acoustic")

    profile = {
        "favorite_genre": genre,
        "favorite_mood": favorite_mood,
        "target_energy": round(target_energy, 3),
        "likes_acoustic": likes_acoustic,
        "avoid_genres": avoid_genres,
    }

    summary = (
        f"Prefers {profile['favorite_genre']} with a {profile['favorite_mood']} vibe "
        f"around energy {profile['target_energy']:.2f}."
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "profile": profile,
        "constraints": constraints,
        "request_summary": summary,
    }


class ProfileParser:
    """Agent 2 that builds recommender-ready user profile preferences."""

    def parse(
        self,
        user_message: str,
        agent1_payload: Dict[str, Any],
        optional_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return parse_profile(
            user_message=user_message,
            agent1_payload=agent1_payload,
            optional_context=optional_context,
            trace_id=trace_id,
        )
