import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _lc_text(response) -> str:
    """Extract plain text from a LangChain response (content may be str or list of blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and "text" in block
        ).strip()
    return str(content).strip()


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


def _token_overlap_confidence(scored: List[Tuple[float, Any]], query_tokens: List[str]) -> float:
    """Confidence proxy for token-overlap: top score normalized by theoretical max."""
    if not scored or not query_tokens:
        return 0.0
    top_score = scored[0][0]
    max_possible = len(query_tokens) + 4.5  # overlap + genre(1.5) + mood(2.0) + tag(1.0)
    return round(min(1.0, top_score / max_possible), 4) if max_possible > 0 else 0.0


_GEMINI_PREFILTER_SIZE = 20  # max songs sent to Gemini; pre-filtered by token overlap


def _parse_gemini_ids_and_confidence(content: str) -> Optional[Tuple[List[Any], float]]:
    """Best-effort parser for Gemini retrieval output.

    Accepts strict JSON, JSON wrapped in extra text, bare arrays, and truncated
    objects that still contain a usable ids prefix (for example: '{"ids": [9, 1').
    """
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        arr_match = re.search(r"\[[\d,\s]+\]", content)
        if arr_match:
            try:
                parsed = {"ids": json.loads(arr_match.group(0)), "confidence": 0.5}
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        # Recover from truncated outputs that still started an ids list.
        # Example: '{"ids": [9, 1' -> ids=[9, 1], confidence defaults to 0.5.
        id_prefix = re.search(r'"ids"\s*:\s*\[([^\]]*)', content)
        if id_prefix:
            raw_items = [item.strip() for item in id_prefix.group(1).split(",")]
            recovered_ids: List[int] = []
            for item in raw_items:
                if not item:
                    continue
                int_match = re.match(r"^-?\d+", item)
                if int_match:
                    recovered_ids.append(int(int_match.group(0)))
            if recovered_ids:
                parsed = {"ids": recovered_ids, "confidence": 0.5}

    if parsed is None:
        return None

    ids = parsed if isinstance(parsed, list) else parsed.get("ids", [])
    confidence = float(parsed.get("confidence", 0.5)) if isinstance(parsed, dict) else 0.5
    confidence = max(0.0, min(1.0, confidence))
    if not isinstance(ids, list):
        return None
    return ids, confidence


def _gemini_retrieve(
    user_message: str,
    songs: List[Dict[str, Any]],
    avoid: set,
    top_n: int,
    api_key: str,
    model: str,
    kb_context: str = "",
) -> Tuple[Optional[Tuple[List[Dict[str, Any]], float]], Optional[str]]:
    """Returns ((ordered_songs, confidence), None) on success, or (None, error_str) on failure."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        eligible = [s for s in songs if str(s.get("genre", "")).lower() not in avoid]

        # Pre-filter: rank by token overlap and keep top GEMINI_PREFILTER_SIZE.
        # This cuts the catalog sent to Gemini from ~40 to ~20 songs, halving prompt tokens.
        query_tokens = _tokenize(user_message)
        if query_tokens and len(eligible) > _GEMINI_PREFILTER_SIZE:
            scored = sorted(eligible, key=lambda s: _score_candidate(query_tokens, s), reverse=True)
            eligible = scored[:_GEMINI_PREFILTER_SIZE]

        catalog = [
            {
                "id": song.get("id"),
                "title": song.get("title"),
                "genre": song.get("genre"),
                "mood": song.get("mood"),
                "mood_tag": song.get("mood_tag"),
            }
            for song in eligible
        ]
        song_by_id = {song.get("id"): song for song in eligible}

        context_block = f"{kb_context}\n\n" if kb_context else ""
        prompt = (
            "You are a music retrieval system. Given a user's vibe request and a song catalog, "
            "return the songs that best match the request semantically.\n\n"
            f"{context_block}"
            f"User request: {user_message!r}\n\n"
            f"Song catalog:\n{json.dumps(catalog, indent=2)}\n\n"
            f"Return ONLY a JSON object with two keys:\n"
            f'  "ids": array of up to {top_n} song IDs ordered from most to least relevant\n'
            f'  "confidence": float 0-1 indicating how well the catalog matches the request\n'
            'Example: {"ids": [3, 7, 1], "confidence": 0.85}\n'
            "No explanation, no markdown, just the JSON object."
        )

        llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0, max_output_tokens=256)
        response = llm.invoke([HumanMessage(content=prompt)])
        content = _lc_text(response)

        parsed = _parse_gemini_ids_and_confidence(content)
        if parsed is None:
            return None, f"could not parse Gemini response as JSON: {content[:120]!r}"

        ids, confidence = parsed

        ordered = [song_by_id[sid] for sid in ids if sid in song_by_id]
        if not ordered:
            return None, "Gemini returned ids but none matched catalog"
        return (ordered[:top_n], round(confidence, 4)), None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _token_overlap_ranked(
    songs: List[Dict[str, Any]],
    avoid: set,
    query_tokens: List[str],
) -> List[Tuple[float, Dict[str, Any]]]:
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for song in songs:
        if str(song.get("genre", "")).lower() in avoid:
            continue
        score = _score_candidate(query_tokens, song)
        scored.append((score, song))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def retrieve_candidates(
    agent2_payload: Dict[str, Any],
    songs: List[Dict[str, Any]],
    top_n: int,
    user_message: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-3-flash-preview",
    kb_docs: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    profile = agent2_payload.get("profile", {}) if isinstance(agent2_payload.get("profile"), dict) else {}
    avoid = {str(genre).lower() for genre in profile.get("avoid_genres", []) if isinstance(genre, str)}
    filtered_out = sum(1 for s in songs if str(s.get("genre", "")).lower() in avoid)
    gemini_fallback_reason: Optional[str] = None

    # Gemini semantic retrieval — understands vibe/intent beyond keyword matching.
    if user_message and api_key:
        kb_context = ""
        kb_docs_injected = 0
        if kb_docs:
            from src.knowledge import retrieve_kb_context, format_kb_context
            profile = agent2_payload.get("profile", {}) if isinstance(agent2_payload.get("profile"), dict) else {}
            relevant = retrieve_kb_context(
                docs=kb_docs,
                genre=profile.get("favorite_genre"),
                mood=profile.get("favorite_mood"),
            )
            kb_context = format_kb_context(relevant)
            kb_docs_injected = len(relevant)

        gemini_result, gemini_error = _gemini_retrieve(
            user_message=user_message,
            songs=songs,
            avoid=avoid,
            top_n=max(1, top_n),
            api_key=api_key,
            model=model,
            kb_context=kb_context,
        )
        if gemini_result:
            gemini_songs, confidence = gemini_result
            backfill_added = 0

            # If Gemini returns fewer than requested ids, top up with token-overlap
            # to satisfy top_n while preserving Gemini ordering first.
            requested_n = max(1, top_n)
            if len(gemini_songs) < requested_n:
                query_tokens = _tokenize(_build_query_text(agent2_payload))
                scored = _token_overlap_ranked(songs=songs, avoid=avoid, query_tokens=query_tokens)
                existing_ids = {song.get("id") for song in gemini_songs}
                for _, song in scored:
                    song_id = song.get("id")
                    if song_id in existing_ids:
                        continue
                    gemini_songs.append(song)
                    existing_ids.add(song_id)
                    backfill_added += 1
                    if len(gemini_songs) >= requested_n:
                        break

            debug = {
                "retriever": "gemini-semantic",
                "query": user_message,
                "candidates_before": len(songs),
                "candidates_after": len(gemini_songs[:requested_n]),
                "filtered_avoid_genres": filtered_out,
                "retrieval_fallback": False,
                "retrieval_confidence": confidence,
                "kb_docs_injected": kb_docs_injected,
            }
            if backfill_added > 0:
                debug["backfill_retriever"] = "token-overlap"
                debug["backfill_added"] = backfill_added
            return gemini_songs[:requested_n], debug
        # Gemini call failed — fall through to token-overlap with error surfaced in debug.
        gemini_fallback_reason = gemini_error or "unknown"

    # Token overlap fallback when Gemini is unavailable.
    query_text = _build_query_text(agent2_payload)
    query_tokens = _tokenize(query_text)

    scored = _token_overlap_ranked(songs=songs, avoid=avoid, query_tokens=query_tokens)

    if not scored:
        fallback = [song for song in songs if str(song.get("genre", "")).lower() not in avoid]
        debug: Dict[str, Any] = {
            "retriever": "token-overlap",
            "query_tokens": query_tokens,
            "candidates_before": len(songs),
            "candidates_after": len(fallback),
            "filtered_avoid_genres": filtered_out,
            "retrieval_fallback": True,
            "retrieval_confidence": 0.0,
        }
        if gemini_fallback_reason:
            debug["gemini_error"] = gemini_fallback_reason
        return fallback[: max(1, top_n)], debug

    selected = [song for _, song in scored[: max(1, top_n)]]
    debug = {
        "retriever": "token-overlap",
        "query_tokens": query_tokens,
        "candidates_before": len(songs),
        "candidates_after": len(selected),
        "filtered_avoid_genres": filtered_out,
        "retrieval_fallback": False,
        "retrieval_confidence": _token_overlap_confidence(scored, query_tokens),
    }
    if gemini_fallback_reason:
        debug["gemini_error"] = gemini_fallback_reason
    return selected, debug
