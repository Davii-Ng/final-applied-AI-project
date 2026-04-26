import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.recommender import recommend_songs
from src.retrieval import retrieve_candidates

SCHEMA_VERSION = "1.0"

_PLAN     = "plan"
_RETRIEVE = "retrieve"
_CHECK    = "check_confidence"
_RETRY    = "retry"
_RANK     = "rank"
_FINALIZE = "finalize"
_DONE     = "done"

CONFIDENCE_RETRY_THRESHOLD = 0.5
MAX_RETRY_ATTEMPTS = 2


@dataclass
class _Step:
    step_name: str
    decision: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"step_name": self.step_name, "decision": self.decision, "data": self.data}


@dataclass
class _State:
    agent2_payload: Dict[str, Any]
    songs: List[Dict[str, Any]]
    k: int
    candidate_pool_size: int
    user_message: Optional[str]
    api_key: Optional[str]
    kb_docs: Optional[List[Dict[str, Any]]]
    trace_id: str
    steps: List[_Step] = field(default_factory=list)
    current_payload: Dict[str, Any] = field(default_factory=dict)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_debug: Dict[str, Any] = field(default_factory=dict)
    ranked: List[Tuple] = field(default_factory=list)
    attempt: int = 0
    confidence: float = 0.0
    retry_triggered: bool = False
    setlist: List[Dict[str, Any]] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)


def _log(state: _State, name: str, decision: str, data: Dict[str, Any]) -> None:
    state.steps.append(_Step(step_name=name, decision=decision, data=data))


def _step_plan(state: _State) -> str:
    profile = state.agent2_payload.get("profile", {})
    _log(state, _PLAN, "analyze_profile_and_initialize", {
        "mood": profile.get("favorite_mood", "balanced"),
        "genre": profile.get("favorite_genre", "pop"),
        "energy": profile.get("target_energy", 0.55),
        "catalog_size": len(state.songs),
        "retry_budget": MAX_RETRY_ATTEMPTS,
        "kb_docs_available": bool(state.kb_docs),
    })
    state.current_payload = state.agent2_payload
    return _RETRIEVE


def _step_retrieve(state: _State) -> str:
    state.attempt += 1
    candidates, debug = retrieve_candidates(
        agent2_payload=state.current_payload,
        songs=state.songs,
        top_n=state.candidate_pool_size,
        user_message=state.user_message,
        api_key=state.api_key,
        kb_docs=state.kb_docs,
    )
    state.candidates = candidates
    state.retrieval_debug = debug
    state.confidence = debug.get("retrieval_confidence", 0.0)

    _log(state, _RETRIEVE, f"tool_call:retrieve_candidates (attempt {state.attempt})", {
        "retriever": debug.get("retriever"),
        "candidates_found": len(candidates),
        "confidence": round(state.confidence, 4),
        "kb_docs_injected": debug.get("kb_docs_injected", 0),
    })
    return _CHECK


def _step_check(state: _State) -> str:
    if state.confidence < CONFIDENCE_RETRY_THRESHOLD and state.attempt < MAX_RETRY_ATTEMPTS:
        _log(state, _CHECK,
             f"confidence={state.confidence:.2f} < threshold={CONFIDENCE_RETRY_THRESHOLD} -> retry",
             {"attempts_used": state.attempt, "budget": MAX_RETRY_ATTEMPTS})
        return _RETRY

    _log(state, _CHECK,
         f"confidence={state.confidence:.2f} -> proceed to rank",
         {"passed_threshold": state.confidence >= CONFIDENCE_RETRY_THRESHOLD})
    return _RANK


def _step_retry(state: _State) -> str:
    state.retry_triggered = True
    broadened = copy.deepcopy(state.current_payload)
    cleared_avoids = broadened.get("profile", {}).get("avoid_genres", [])
    broadened["profile"]["avoid_genres"] = []
    new_pool = state.candidate_pool_size * 2

    _log(state, _RETRY, "broaden_search:relax_genre_constraints", {
        "cleared_avoid_genres": cleared_avoids,
        "old_pool_size": state.candidate_pool_size,
        "new_pool_size": new_pool,
        "reason": f"confidence {state.confidence:.2f} below threshold {CONFIDENCE_RETRY_THRESHOLD}",
    })
    state.current_payload = broadened
    state.candidate_pool_size = new_pool
    return _RETRIEVE


def _step_rank(state: _State) -> str:
    profile = state.agent2_payload.get("profile", {})
    user_prefs: Dict[str, Any] = {
        "genre": profile.get("favorite_genre", "pop"),
        "mood": profile.get("favorite_mood", "balanced"),
        "energy": profile.get("target_energy", 0.55),
        "likes_acoustic": bool(profile.get("likes_acoustic", False)),
    }
    for key in ("target_valence", "target_danceability", "target_tempo_bpm",
                "target_acousticness", "target_instrumentalness", "target_brightness"):
        if key in profile:
            user_prefs[key] = profile[key]

    state.ranked = recommend_songs(user_prefs=user_prefs, songs=state.candidates, k=state.k)
    top_score = round(state.ranked[0][1], 4) if state.ranked else 0.0

    _log(state, _RANK, "tool_call:recommend_songs -> cosine_similarity_scoring", {
        "candidates_scored": len(state.candidates),
        "k_requested": state.k,
        "k_returned": len(state.ranked),
        "top_score": top_score,
    })
    return _FINALIZE


def _step_finalize(state: _State) -> str:
    for idx, (song, score, reasons) in enumerate(state.ranked, start=1):
        state.setlist.append({
            "rank": idx,
            "title": song.get("title", ""),
            "artist": song.get("artist", ""),
            "score": round(float(score), 4),
        })
        state.explanations.append(
            "; ".join(reasons) if isinstance(reasons, list) else str(reasons)
        )

    _log(state, _FINALIZE, "setlist_assembled", {
        "k": len(state.setlist),
        "retry_triggered": state.retry_triggered,
    })
    return _DONE


_TRANSITIONS = {
    _PLAN:     _step_plan,
    _RETRIEVE: _step_retrieve,
    _CHECK:    _step_check,
    _RETRY:    _step_retry,
    _RANK:     _step_rank,
    _FINALIZE: _step_finalize,
}


def _run_workflow(state: _State) -> _State:
    current = _PLAN
    while current != _DONE:
        current = _TRANSITIONS[current](state)
    return state


def _build_user_prefs(profile: Dict[str, Any]) -> Dict[str, Any]:
    prefs: Dict[str, Any] = {
        "genre": profile.get("favorite_genre", "pop"),
        "mood": profile.get("favorite_mood", "balanced"),
        "energy": profile.get("target_energy", 0.55),
        "likes_acoustic": bool(profile.get("likes_acoustic", False)),
    }
    for key in ("target_valence", "target_danceability", "target_tempo_bpm",
                "target_acousticness", "target_instrumentalness", "target_brightness"):
        if key in profile:
            prefs[key] = profile[key]
    return prefs


def curate_setlist(
    agent2_payload: Dict[str, Any],
    songs: List[Dict[str, Any]],
    k: int = 5,
    candidate_pool_size: int = 20,
    trace_id: Optional[str] = None,
    user_message: Optional[str] = None,
    api_key: Optional[str] = None,
    kb_docs: Optional[List[Dict[str, Any]]] = None,
    agentic: bool = True,
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
            "retrieval": {},
            "agentic_steps": [],
            "retry_triggered": False,
            "error": "invalid_profile_payload",
        }

    k_int = max(1, int(k))
    pool_size = max(k_int, int(candidate_pool_size))

    if agentic:
        state = _State(
            agent2_payload=agent2_payload,
            songs=songs,
            k=k_int,
            candidate_pool_size=pool_size,
            user_message=user_message,
            api_key=api_key,
            kb_docs=kb_docs,
            trace_id=resolved_trace,
        )
        _run_workflow(state)
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": resolved_trace,
            "setlist": state.setlist,
            "explanations": state.explanations,
            "profile_echo": profile,
            "retrieval": state.retrieval_debug,
            "agentic_steps": [s.to_dict() for s in state.steps],
            "retry_triggered": state.retry_triggered,
        }

    # Simple (non-agentic) path: single retrieve + rank, no state machine overhead.
    candidates, retrieval_debug = retrieve_candidates(
        agent2_payload=agent2_payload,
        songs=songs,
        top_n=pool_size,
        user_message=user_message,
        api_key=api_key,
        kb_docs=kb_docs,
    )
    ranked = recommend_songs(user_prefs=_build_user_prefs(profile), songs=candidates, k=k_int)

    setlist: List[Dict[str, Any]] = []
    explanations: List[str] = []
    for idx, (song, score, reasons) in enumerate(ranked, start=1):
        setlist.append({
            "rank": idx,
            "title": song.get("title", ""),
            "artist": song.get("artist", ""),
            "score": round(float(score), 4),
        })
        explanations.append("; ".join(reasons) if isinstance(reasons, list) else str(reasons))

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "setlist": setlist,
        "explanations": explanations,
        "profile_echo": profile,
        "retrieval": retrieval_debug,
    }


class SetlistCurator:
    def curate(
        self,
        agent2_payload: Dict[str, Any],
        songs: List[Dict[str, Any]],
        k: int = 5,
        candidate_pool_size: int = 20,
        trace_id: Optional[str] = None,
        user_message: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_docs: Optional[List[Dict[str, Any]]] = None,
        agentic: bool = True,
    ) -> Dict[str, Any]:
        return curate_setlist(
            agent2_payload=agent2_payload,
            songs=songs,
            k=k,
            candidate_pool_size=candidate_pool_size,
            trace_id=trace_id,
            user_message=user_message,
            api_key=api_key,
            kb_docs=kb_docs,
            agentic=agentic,
        )
