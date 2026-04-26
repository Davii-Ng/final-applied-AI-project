import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

SCHEMA_VERSION = "1.0"
FALLBACK_MOOD = "balanced"
CONFIDENCE_FALLBACK_THRESHOLD = 0.55

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
        "tired": 0.8,
        "exhausted": 0.9,
        "drained": 0.9,
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
        "mad": 1.0,
        "angry": 1.1,
        "frustrated": 1.0,
        "annoyed": 0.9,
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
    "i'm mad": ("moody", 2.0),
    "im mad": ("moody", 2.0),
    "so mad": ("moody", 1.8),
    "feeling tired": ("relaxed", 1.5),
    "bit tired": ("relaxed", 1.5),
    "so tired": ("relaxed", 1.5),
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
        "I am tired and exhausted, I need something calming and low energy to wind down",
    ],
    "moody": [
        "I am feeling dark brooding atmospheric and introspective, late night deep thoughts",
        "moody dark atmospheric cloudy brooding melancholic night drive indie alternative",
        "complex emotional music, bittersweet, contemplative, rainy window, introspective",
        "I am angry and frustrated, feeling mad and annoyed, intense dark emotions",
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
        "time to hit the gym, I need motivation and power to crush this workout",
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

# Minimum cosine similarity to accept best mood rather than falling back to balanced.
# MiniLM-L6-v2 averaged-reference scores top out around 0.45 for strong matches;
# 0.20 correctly rejects truly empty/ambiguous inputs while accepting short real phrases.
_ST_CONFIDENCE_THRESHOLD = 0.20


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

    # Hybrid boost: phrase/keyword signals handle short colloquial inputs that
    # confuse pure cosine similarity (e.g. "A bit tired" → relaxed, "I'm mad" → moody).
    # Keyword scores are raw counts; we normalize by a typical max (~3.0) and add
    # a small fraction so they can tip close calls without dominating clear ST wins.
    tokens = _tokenize(user_message)
    keyword_scores = _score_moods(tokens, user_message)
    max_kw = max(keyword_scores.values()) if keyword_scores else 0.0
    if max_kw > 0:
        for i, mood in enumerate(moods):
            kw = keyword_scores.get(mood, 0.0)
            # Scale keyword contribution: 0.08 at max typical score (~3.0), capped at 0.10
            mood_scores[i] += min(kw / max_kw * 0.08, 0.10)

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
) -> Dict[str, Any]:
    selected_backend = (backend or "local").strip().lower()

    if selected_backend == "sentence_transformers":
        return _st_analyze_mood(user_message, optional_context, trace_id)

    if selected_backend == "local":
        return _local_analyze_mood(user_message, optional_context, trace_id)

    raise ValueError("backend must be one of: local, sentence_transformers")


class MoodAnalyst:
    """Agent 1 — converts free text into a normalized mood payload using sentence-transformers or local keyword scoring."""

    def analyze(
        self,
        user_message: str,
        optional_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        backend: str = "local",
    ) -> Dict[str, Any]:
        return analyze_mood(
            user_message=user_message,
            optional_context=optional_context,
            trace_id=trace_id,
            backend=backend,
        )
