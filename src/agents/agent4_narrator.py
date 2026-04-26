import os
import re
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


def _lc_text(response) -> str:
    """Extract plain text from a LangChain response (content may be str or list of blocks)."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and "text" in block
        ).strip()
    return str(content).strip()


def _gemini_paragraph(
    mood: str,
    genre: str,
    energy: float,
    setlist: List[Dict[str, Any]],
    api_key: str,
    model: str = "gemini-3-flash-preview",
) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    tracks = ", ".join(
        f"{item.get('title', '?')} by {item.get('artist', '?')}" for item in setlist
    )
    energy_label = "high" if energy >= 0.75 else "medium" if energy >= 0.5 else "low"

    prompt = (
        "You are a DJ narrator for a music recommender app. "
        "Write a short, natural 2-3 sentence paragraph introducing this playlist. "
        "Mention the vibe and reference the songs by name. Keep it conversational and engaging. "
        "Do not use bullet points or headers — just a flowing paragraph.\n\n"
        f"Mood: {mood} | Genre: {genre} | Energy: {energy_label}\n"
        f"Tracks: {tracks}"
    )

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.4, max_output_tokens=150)
    response = llm.invoke([HumanMessage(content=prompt)])
    return _lc_text(response)


def _template_paragraph(mood: str, genre: str, setlist: List[Dict[str, Any]]) -> str:
    titles = [item.get("title", "Unknown") for item in setlist]
    tracks_text = ", ".join(titles[:-1]) + (f" and {titles[-1]}" if len(titles) > 1 else titles[0] if titles else "")
    return (
        f"Tonight's mix leans {mood} with a {genre} center. "
        f"The lineup — {tracks_text} — is built to keep the energy coherent from start to finish. "
        "Tell me what to push more if you want a second pass."
    )


def narrate_setlist(
    agent3_payload: Dict[str, Any],
    persona: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    backend: str = "local",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_trace = trace_id or str(agent3_payload.get("trace_id", "")).strip() or "trace-missing"

    setlist = agent3_payload.get("setlist", [])
    explanations = agent3_payload.get("explanations", [])
    profile = agent3_payload.get("profile_echo", {})

    if not isinstance(setlist, list) or not setlist:
        fallback_paragraph = "I couldn't build a full setlist yet. Give me one more detail — genre, mood, or energy — and I'll refine the mix."
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": resolved_trace,
            "intro": fallback_paragraph,
            "track_transitions": [],
            "closing": "",
            "paragraph": fallback_paragraph,
            "safety_notes": ["empty_setlist_fallback"],
        }

    mood = profile.get("favorite_mood", "balanced") if isinstance(profile, dict) else "balanced"
    genre = profile.get("favorite_genre", "mixed") if isinstance(profile, dict) else "mixed"
    energy = float(profile.get("target_energy", 0.55)) if isinstance(profile, dict) else 0.55

    # Build paragraph — try Gemini, fall back to template
    paragraph = ""
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if backend in {"gemini", "auto"} and resolved_key:
        try:
            paragraph = _gemini_paragraph(mood, genre, energy, setlist, resolved_key)
        except Exception:
            paragraph = ""

    if not paragraph:
        paragraph = _template_paragraph(mood, genre, setlist)

    # Keep legacy fields for backward compatibility
    intro = f"Tonight's mix leans {mood} with a {genre} center. I built this sequence to keep energy coherent without repeating too much."

    transitions: List[str] = []
    style = (persona or {}).get("style", "friendly")
    for idx, item in enumerate(setlist):
        title = item.get("title", "Unknown Track")
        artist = item.get("artist", "Unknown Artist")
        rank = item.get("rank", idx + 1)
        reason = explanations[idx] if idx < len(explanations) else "selected for overall fit"
        transitions.append(f"#{rank}: {title} by {artist} - {reason}.")

    if style == "concise":
        transitions = transitions[: max(1, min(3, len(transitions)))]

    return {
        "schema_version": SCHEMA_VERSION,
        "trace_id": resolved_trace,
        "intro": intro,
        "track_transitions": transitions,
        "closing": "If you want a second pass, tell me what to push more: energy, mood, or genre diversity.",
        "paragraph": paragraph,
        "safety_notes": [],
    }


class DJNarrator:
    """Agent 4 that transforms ranked setlists into DJ narration text."""

    def narrate(
        self,
        agent3_payload: Dict[str, Any],
        persona: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        backend: str = "local",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        return narrate_setlist(
            agent3_payload=agent3_payload,
            persona=persona,
            trace_id=trace_id,
            backend=backend,
            api_key=api_key,
        )
