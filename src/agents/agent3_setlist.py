from typing import Any, Dict, List, Optional

from src.recommender import recommend_songs
from src.retrieval import retrieve_candidates

SCHEMA_VERSION = "1.0"


def _build_user_prefs(profile_payload: Dict[str, Any]) -> Dict[str, Any]:
    profile = profile_payload.get("profile", {}) if isinstance(profile_payload.get("profile"), dict) else {}
    return {
        "genre": profile.get("favorite_genre", "pop"),
        "mood": profile.get("favorite_mood", "balanced"),
        "energy": profile.get("target_energy", 0.55),
        "likes_acoustic": bool(profile.get("likes_acoustic", False)),
    }


def curate_setlist(
    agent2_payload: Dict[str, Any],
    songs: List[Dict[str, Any]],
    k: int = 5,
    candidate_pool_size: int = 20,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_trace = trace_id or str(agent2_payload.get("trace_id", "")).strip() or "trace-missing"

    profile = agent2_payload.get("profile")
    if not isinstance(profile, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": resolved_trace,
            "setlist": [],
            "explanations": [],
            "profile_echo": {},
            "error": "invalid_profile_payload",
        }

    user_prefs = _build_user_prefs(agent2_payload)
    k_int = max(1, int(k))
    pool_size = max(k_int, int(candidate_pool_size))

    retrieved_candidates, retrieval_debug = retrieve_candidates(
        agent2_payload=agent2_payload,
        songs=songs,
        top_n=pool_size,
    )

    ranked = recommend_songs(user_prefs=user_prefs, songs=retrieved_candidates, k=k_int)

    setlist: List[Dict[str, Any]] = []
    explanations: List[str] = []
    for index, (song, score, reasons) in enumerate(ranked, start=1):
        setlist.append(
            {
                "rank": index,
                "title": song.get("title", ""),
                "artist": song.get("artist", ""),
                "score": round(float(score), 4),
            }
        )
        reason_text = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)
        explanations.append(reason_text)

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "setlist": setlist,
        "explanations": explanations,
        "profile_echo": profile,
        "retrieval": retrieval_debug,
    }


class SetlistCurator:
    """Agent 3 that turns profile payloads into ranked setlists."""

    def curate(
        self,
        agent2_payload: Dict[str, Any],
        songs: List[Dict[str, Any]],
        k: int = 5,
        candidate_pool_size: int = 20,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return curate_setlist(
            agent2_payload=agent2_payload,
            songs=songs,
            k=k,
            candidate_pool_size=candidate_pool_size,
            trace_id=trace_id,
        )
