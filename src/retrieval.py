import re
from typing import Any, Dict, List, Tuple


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _song_text(song: Dict[str, Any]) -> str:
    fields = [
        str(song.get("title", "")),
        str(song.get("artist", "")),
        str(song.get("genre", "")),
        str(song.get("mood", "")),
        str(song.get("mood_tag", "")),
    ]
    return " ".join(fields)


def _build_query_text(agent2_payload: Dict[str, Any]) -> str:
    profile = agent2_payload.get("profile", {}) if isinstance(agent2_payload.get("profile"), dict) else {}
    constraints = agent2_payload.get("constraints", {}) if isinstance(agent2_payload.get("constraints"), dict) else {}

    parts = [
        str(agent2_payload.get("request_summary", "")),
        str(profile.get("favorite_genre", "")),
        str(profile.get("favorite_mood", "")),
        "acoustic" if bool(profile.get("likes_acoustic", False)) else "",
        " ".join(str(item) for item in profile.get("avoid_genres", []) if isinstance(item, str)),
        " ".join(str(item) for item in constraints.get("inferred_fields", []) if isinstance(item, str)),
    ]
    return " ".join(parts)


def _score_candidate(query_tokens: List[str], song: Dict[str, Any]) -> float:
    text = _song_text(song)
    tokens = set(_tokenize(text))
    if not tokens:
        return 0.0

    overlap = sum(1 for token in query_tokens if token in tokens)
    genre_boost = 1.5 if str(song.get("genre", "")).lower() in query_tokens else 0.0
    mood_boost = 2.0 if str(song.get("mood", "")).lower() in query_tokens else 0.0
    tag_boost = 1.0 if str(song.get("mood_tag", "")).lower() in query_tokens else 0.0

    return overlap + genre_boost + mood_boost + tag_boost


def retrieve_candidates(
    agent2_payload: Dict[str, Any],
    songs: List[Dict[str, Any]],
    top_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    profile = agent2_payload.get("profile", {}) if isinstance(agent2_payload.get("profile"), dict) else {}
    avoid = {str(genre).lower() for genre in profile.get("avoid_genres", []) if isinstance(genre, str)}

    query_text = _build_query_text(agent2_payload)
    query_tokens = _tokenize(query_text)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    filtered_out = 0
    for song in songs:
        if str(song.get("genre", "")).lower() in avoid:
            filtered_out += 1
            continue
        score = _score_candidate(query_tokens, song)
        scored.append((score, song))

    # If retrieval is weak, fall back to full catalog minus explicit avoids.
    if not scored:
        fallback = [song for song in songs if str(song.get("genre", "")).lower() not in avoid]
        debug = {
            "retriever": "rag-lite",
            "query_tokens": query_tokens,
            "candidates_before": len(songs),
            "candidates_after": len(fallback),
            "filtered_avoid_genres": filtered_out,
            "retrieval_fallback": True,
        }
        return fallback[: max(1, top_n)], debug

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [song for _, song in scored[: max(1, top_n)]]
    debug = {
        "retriever": "rag-lite",
        "query_tokens": query_tokens,
        "candidates_before": len(songs),
        "candidates_after": len(selected),
        "filtered_avoid_genres": filtered_out,
        "retrieval_fallback": False,
    }
    return selected, debug
