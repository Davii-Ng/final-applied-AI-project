"""Interactive CLI for the rebuilt 4-agent recommendation pipeline."""

import os
import sys
from textwrap import wrap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""} and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestrator import run_pipeline
from src.recommender import load_songs


def _print_section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def _prompt_top_k(default_k: int = 3) -> int:
    raw = input(f"How many recommendations? [default {default_k}]: ").strip()
    if not raw:
        return default_k
    try:
        return max(1, min(int(raw), 20))
    except ValueError:
        print(f"Invalid number. Using default {default_k}.")
        return default_k


def _prompt_backend(default_backend: str = "auto") -> str:
    raw = input(f"Agent1 backend local/gemini/auto [default {default_backend}]: ").strip().lower()
    if raw in {"local", "gemini", "auto"}:
        return raw
    if raw:
        print(f"Invalid backend. Using {default_backend}.")
    return default_backend


def _prompt_output_mode(default_mode: str = "compact") -> str:
    raw = input(f"Output mode compact/verbose [default {default_mode}]: ").strip().lower()
    if raw in {"compact", "verbose"}:
        return raw
    if raw:
        print(f"Invalid mode. Using {default_mode}.")
    return default_mode


def _print_setlist(setlist: list[dict], explanations: list[str]) -> None:
    if not setlist:
        print("No setlist available.")
        return

    rank_width = 4
    title_width = 24
    artist_width = 18
    score_width = 8
    why_width = 56

    line = "+" + "-" * (rank_width + 2)
    line += "+" + "-" * (title_width + 2)
    line += "+" + "-" * (artist_width + 2)
    line += "+" + "-" * (score_width + 2)
    line += "+" + "-" * (why_width + 2) + "+"

    print("\nsetlist:")
    print(line)
    print(
        f"| {'#':<{rank_width}} | {'Title':<{title_width}} | {'Artist':<{artist_width}} | {'Score':>{score_width}} | {'Why':<{why_width}} |"
    )
    print(line)

    for index, item in enumerate(setlist):
        rank = int(item.get("rank", index + 1))
        title = str(item.get("title", ""))
        artist = str(item.get("artist", ""))
        score = float(item.get("score", 0.0))
        reason = explanations[index] if index < len(explanations) else "selected for overall fit"
        reason_head = reason.split(";")[0].strip() or "selected for overall fit"
        wrapped = wrap(reason_head, width=why_width) or [""]

        print(
            f"| {rank:<{rank_width}} | {title[:title_width]:<{title_width}} | {artist[:artist_width]:<{artist_width}} | {score:>{score_width}.2f} | {wrapped[0]:<{why_width}} |"
        )
        for cont in wrapped[1:]:
            print(
                f"| {'':<{rank_width}} | {'':<{title_width}} | {'':<{artist_width}} | {'':>{score_width}} | {cont:<{why_width}} |"
            )

    print(line)


def _print_pipeline_result(result: dict, output_mode: str) -> None:
    agent1 = result.get("agent1", {})
    agent2 = result.get("agent2", {})
    agent3 = result.get("agent3", {})
    agent4 = result.get("agent4", {})

    profile = agent2.get("profile", {}) if isinstance(agent2.get("profile"), dict) else {}
    constraints = agent2.get("constraints", {}) if isinstance(agent2.get("constraints"), dict) else {}
    setlist = agent3.get("setlist", []) if isinstance(agent3.get("setlist"), list) else []
    explanations = agent3.get("explanations", []) if isinstance(agent3.get("explanations"), list) else []
    retrieval = agent3.get("retrieval", {}) if isinstance(agent3.get("retrieval"), dict) else {}

    print(f"trace_id: {result.get('trace_id')}")
    print(
        "agent1: "
        f"mood={agent1.get('detected_mood')} | confidence={agent1.get('confidence')} | "
        f"energy_hint={agent1.get('energy_hint')}"
    )
    print(
        "agent2: "
        f"genre={profile.get('favorite_genre')} | mood={profile.get('favorite_mood')} | "
        f"energy={profile.get('target_energy')} | likes_acoustic={profile.get('likes_acoustic')}"
    )
    print(
        "agent2 constraints: "
        f"missing={constraints.get('missing_fields', [])} | inferred={constraints.get('inferred_fields', [])}"
    )
    print(
        "retrieval: "
        f"mode={retrieval.get('retriever', 'none')} | "
        f"pool={retrieval.get('candidates_after', 'n/a')} | "
        f"avoids_filtered={retrieval.get('filtered_avoid_genres', 0)}"
    )

    _print_setlist(setlist=setlist, explanations=explanations)

    print("\nnarration:")
    print(f"- intro: {agent4.get('intro', '')}")
    transitions = agent4.get("track_transitions", []) if isinstance(agent4.get("track_transitions"), list) else []
    for transition in transitions:
        print(f"- {transition}")
    print(f"- closing: {agent4.get('closing', '')}")

    if output_mode == "verbose":
        print("\nraw payloads:")
        print(f"agent1: {agent1}")
        print(f"agent2: {agent2}")
        print(f"agent3: {agent3}")
        print(f"agent4: {agent4}")


def main() -> None:
    songs_csv = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(songs_csv)

    _print_section("Interactive Multi-Agent DJ Recommender")
    print("Flow: Agent1 -> Agent2 -> Agent3 -> Agent4")
    print("Type 'exit' to quit.")

    k = _prompt_top_k(default_k=3)
    backend = _prompt_backend(default_backend="gemini")
    output_mode = _prompt_output_mode(default_mode="compact")

    while True:
        try:
            message = input("\nDescribe your vibe: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive CLI.")
            return

        if message.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            return

        if not message:
            print("Please enter a non-empty request.")
            continue

        result = run_pipeline(user_message=message, songs=songs, k=k, agent1_backend=backend)
        _print_pipeline_result(result, output_mode=output_mode)


if __name__ == "__main__":
    main()
