import os
from typing import Any, Dict, List, Optional

from src.agents.agent1_mood import analyze_mood
from src.agents.agent2_profile import parse_profile
from src.agents.agent4_narrator import narrate_setlist


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run_pipeline(
    user_message: str,
    songs: List[Dict[str, Any]],
    k: int = 5,
    agent1_backend: str = "auto",
    agent1_model: str = "gemini-3-flash-preview",
    agent1_api_key: Optional[str] = None,
    agent4_backend: str = "gemini",
    agent4_api_key: Optional[str] = None,
    optional_context: Optional[Dict[str, Any]] = None,
    persona: Optional[Dict[str, Any]] = None,
    use_agentic: bool = False,
    kb_docs: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    if verbose:
        _log(f"[Agent 1] Detecting mood (sentence-transformers)...")
    agent1_payload = analyze_mood(
        user_message=user_message,
        optional_context=optional_context,
        backend=agent1_backend,
        model=agent1_model,
        api_key=agent1_api_key,
    )

    trace_id = agent1_payload.get("trace_id")

    if verbose:
        mood = agent1_payload.get("detected_mood", "?")
        notes = agent1_payload.get("notes", "")
        if agent1_payload.get("llm_profile"):
            backend_used = "gemini"
        elif "sentence-transformer" in notes:
            backend_used = "sentence-transformers"
        elif "gemini fallback" in notes:
            backend_used = "local (gemini failed)"
        else:
            backend_used = "local"
        _log(f"[Agent 1] Done  — mood={mood} (via {backend_used})")
        _log(f"[Agent 2] Building listener profile...")

    agent2_payload = parse_profile(
        user_message=user_message,
        agent1_payload=agent1_payload,
        optional_context=optional_context,
        trace_id=trace_id,
    )

    resolved_api_key = (
        agent4_api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )
    if verbose and not resolved_api_key:
        _log("[warn] No API key found — set GOOGLE_API_KEY or GEMINI_API_KEY in .env to enable Gemini")

    from src.agents.agent3 import curate_setlist as _curate

    retriever_label = "gemini-semantic" if resolved_api_key else "token-overlap"
    mode_label = "agentic" if use_agentic else "standard"
    if verbose:
        _log(f"[Agent 2] Done  — genre={agent2_payload.get('profile', {}).get('favorite_genre', '?')}")
        _log(f"[Agent 3] Curating setlist ({mode_label}, retriever={retriever_label})...")

    agent3_payload = _curate(
        agent2_payload=agent2_payload,
        songs=songs,
        k=k,
        trace_id=trace_id,
        user_message=user_message,
        api_key=resolved_api_key,
        kb_docs=kb_docs,
        agentic=use_agentic,
    )

    if verbose:
        n = len(agent3_payload.get("setlist", []))
        actual_retriever = agent3_payload.get("retrieval", {}).get("retriever", retriever_label)
        _log(f"[Agent 3] Done  — {n} tracks (retriever={actual_retriever})")
        _log(f"[Agent 4] Writing narration ({agent4_backend})...")

    agent4_payload = narrate_setlist(
        agent3_payload=agent3_payload,
        persona=persona,
        trace_id=trace_id,
        backend=agent4_backend,
        api_key=resolved_api_key,
    )

    if verbose:
        _log(f"[Agent 4] Done  — narration ready")

    result: Dict[str, Any] = {
        "trace_id": trace_id,
        "agent1": agent1_payload,
        "agent2": agent2_payload,
        "agent3": agent3_payload,
        "agent4": agent4_payload,
    }
    if use_agentic and "agentic_steps" in agent3_payload:
        result["agentic_steps"] = agent3_payload["agentic_steps"]
    return result
