"""Interactive CLI for the 4-agent DJ music recommender."""

import os
import re
import sys
from textwrap import wrap

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""} and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestrator import run_pipeline
from src.recommender import load_songs
from src.knowledge import load_knowledge_base


def _prompt_top_k(default_k: int = 3) -> int:
    raw = input(f"How many songs? [default {default_k}]: ").strip()
    if not raw:
        return default_k
    try:
        return max(1, min(int(raw), 20))
    except ValueError:
        print(f"  Invalid number, using {default_k}.")
        return default_k


def _energy_label(energy: float) -> str:
    if energy >= 0.75:
        return "high"
    if energy >= 0.5:
        return "medium"
    return "low"


def _top_reason(reason_str: str) -> str:
    parts = [r.strip() for r in reason_str.split(";") if r.strip()]

    def _score(r: str) -> float:
        m = re.search(r"\(\+([0-9.]+)\)", r)
        return float(m.group(1)) if m else 0.0

    return max(parts, key=_score, default="selected for overall fit")


def _print_setlist(setlist: list, explanations: list) -> None:
    if not setlist:
        print("  No recommendations found.")
        return

    title_width = 24
    artist_width = 20
    why_width = 30

    header = f"  {'#':<3}  {'Title':<{title_width}}  {'Artist':<{artist_width}}  {'Why':<{why_width}}"
    divider = "  " + "─" * (3 + 2 + title_width + 2 + artist_width + 2 + why_width)

    print(header)
    print(divider)

    for index, item in enumerate(setlist):
        rank = int(item.get("rank", index + 1))
        title = str(item.get("title", ""))[:title_width]
        artist = str(item.get("artist", ""))[:artist_width]
        reason_raw = explanations[index] if index < len(explanations) else "selected for overall fit"
        why = _top_reason(reason_raw)[:why_width]

        print(f"  {rank:<3}  {title:<{title_width}}  {artist:<{artist_width}}  {why}")


def _print_agentic_steps(steps: list, retry_triggered: bool) -> None:
    if not steps:
        return

    step_label_width = 10
    print("  Reasoning Steps:")
    for step in steps:
        name     = step.get("step_name", "?")
        decision = step.get("decision", "")
        data     = step.get("data", {})

        extras = []
        if "confidence" in data:
            extras.append(f"conf={data['confidence']:.2f}")
        if "retriever" in data:
            extras.append(f"via={data['retriever']}")
        if "candidates_found" in data:
            extras.append(f"found={data['candidates_found']}")
        if "top_score" in data:
            extras.append(f"top={data['top_score']:.2f}")
        if "kb_docs_injected" in data and data["kb_docs_injected"]:
            extras.append(f"kb={data['kb_docs_injected']}")
        if "cleared_avoid_genres" in data and data["cleared_avoid_genres"]:
            extras.append(f"cleared={data['cleared_avoid_genres']}")

        suffix = "  " + "  ".join(extras) if extras else ""
        label  = f"[{name}]"
        print(f"  {label:<{step_label_width}}  {decision}{suffix}")

    if retry_triggered:
        print("  * Retry was triggered — search was broadened for better results")
    print()


def _print_result(result: dict) -> None:
    agent1 = result.get("agent1", {})
    agent2 = result.get("agent2", {})
    agent3 = result.get("agent3", {})
    agent4 = result.get("agent4", {})

    profile = agent2.get("profile", {}) if isinstance(agent2.get("profile"), dict) else {}
    setlist = agent3.get("setlist", []) if isinstance(agent3.get("setlist"), list) else []
    explanations = agent3.get("explanations", []) if isinstance(agent3.get("explanations"), list) else []

    mood = profile.get("favorite_mood") or agent1.get("detected_mood") or "balanced"
    genre = profile.get("favorite_genre") or "mixed"
    energy = float(profile.get("target_energy") or agent1.get("energy_hint") or 0.55)

    print()
    print(f"  Mood: {mood:<12}  Energy: {_energy_label(energy):<8}  Genre: {genre}")
    print()

    retrieval_debug = agent3.get("retrieval", {})
    if retrieval_debug.get("gemini_error"):
        print(f"  [warn] Gemini retrieval failed (using token-overlap): {retrieval_debug['gemini_error']}")
        print()

    agentic_steps = result.get("agentic_steps") or agent3.get("agentic_steps", [])
    retry_triggered = agent3.get("retry_triggered", False)
    if agentic_steps:
        _print_agentic_steps(agentic_steps, retry_triggered)

    _print_setlist(setlist, explanations)

    paragraph = agent4.get("paragraph", "")
    if paragraph:
        print()
        for line in wrap(paragraph, width=72):
            print(f"  {line}")

    print()


def main() -> None:
    songs_csv = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(songs_csv)

    kb_path = os.path.join(PROJECT_ROOT, "data", "knowledge_base.json")
    kb_docs = load_knowledge_base(kb_path) if os.path.exists(kb_path) else None

    print("\n  DJ Recommender  [agentic mode]")
    print("  " + "-" * 40)
    k = _prompt_top_k(default_k=3)
    print()
    print("  Type your vibe. Enter 'quit' to exit.")

    while True:
        try:
            message = input("\nDescribe your vibe: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            return

        if message.lower() in {"exit", "quit", "q"}:
            print("  Goodbye.")
            return

        if not message:
            print("  Say something -- anything about your mood or what you're doing.")
            continue

        result = run_pipeline(
            user_message=message,
            songs=songs,
            k=k,
            agent4_backend="gemini",
            use_agentic=True,
            kb_docs=kb_docs,
            verbose=True,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
