"""
Command line runner for the Music Recommender Simulation.

This CLI shows the current project state:
- Deterministic recommendation scoring in src.recommender
- Agent 1 mood parsing in src.agents.agent1_mood
- A lightweight bridge from user text to recommender-ready prefs
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Support running this file directly from the src folder.
if __package__ in {None, ""}:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from src.recommender import load_songs, recommend_songs
from src.agents.agent1_mood import analyze_mood
from textwrap import wrap


def _print_section(title: str) -> None:
    rule = "=" * len(title)
    print(f"\n{title}\n{rule}")


def _format_prefs(user_prefs: dict) -> str:
    keys = ["genre", "mood", "energy", "likes_acoustic"]
    parts = []
    for key in keys:
        if key in user_prefs:
            parts.append(f"{key}={user_prefs[key]}")
    return " | ".join(parts)


def _print_agent_payload(payload: dict) -> None:
    confidence = payload.get("confidence")
    confidence_text = f"{confidence:.2f}" if isinstance(confidence, (float, int)) else str(confidence)
    candidates = payload.get("mood_candidates", [])
    candidates_text = ", ".join(candidates) if isinstance(candidates, list) and candidates else "none"
    fallback_used = payload.get("detected_mood") == "balanced" and confidence is not None and confidence < 0.55

    print("agent payload summary:")
    print(f"  trace_id: {payload.get('trace_id', 'missing')}")
    print(f"  detected_mood: {payload.get('detected_mood', 'missing')}")
    print(f"  confidence: {confidence_text}")
    print(f"  energy_hint: {payload.get('energy_hint')}")
    print(f"  mood_candidates: {candidates_text}")
    print(f"  fallback_used: {'yes' if fallback_used else 'no'}")
    print(f"  notes: {payload.get('notes', 'missing')}")


def _print_recommendation_table(profile_name: str, user_prefs: dict, recommendations) -> None:
    rank_width = 4
    title_width = 24
    artist_width = 18
    score_width = 7
    reasons_width = 56

    line = "+" + "-" * (rank_width + 2)
    line += "+" + "-" * (title_width + 2)
    line += "+" + "-" * (artist_width + 2)
    line += "+" + "-" * (score_width + 2)
    line += "+" + "-" * (reasons_width + 2) + "+"

    print(f"=== {profile_name} ===")
    print(f"prefs: {_format_prefs(user_prefs)}")
    print()
    print(line)
    print(
        f"| {'#':<{rank_width}} | {'Title':<{title_width}} | {'Artist':<{artist_width}} | {'Score':>{score_width}} | {'Reasons':<{reasons_width}} |"
    )
    print(line)

    for index, (song, score, reasons) in enumerate(recommendations, start=1):
        reason_text = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)
        wrapped_reasons = wrap(reason_text, width=reasons_width) or [""]
        first_line = wrapped_reasons[0]
        print(
            f"| {index:<{rank_width}} | {song['title'][:title_width]:<{title_width}} | {song['artist'][:artist_width]:<{artist_width}} | {score:>{score_width}.2f} | {first_line:<{reasons_width}} |"
        )
        for continuation in wrapped_reasons[1:]:
            print(
                f"| {'':<{rank_width}} | {'':<{title_width}} | {'':<{artist_width}} | {'':>{score_width}} | {continuation:<{reasons_width}} |"
            )

    print(line)
    print()


def _prefs_from_agent_message(message: str, default_genre: str = "pop") -> tuple[dict, dict]:
    payload = analyze_mood(message)
    mood = payload["detected_mood"]
    energy = payload["energy_hint"] if payload["energy_hint"] is not None else 0.55
    likes_acoustic = mood in {"chill", "relaxed", "sad", "focused", "nostalgic", "moody"}

    user_prefs = {
        "genre": default_genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }
    return user_prefs, payload


def main() -> None:
    songs_csv = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(songs_csv)

    _print_section("Music Recommender Simulation")
    print("Current focus: Day 1 is locked in, Agent 1 is complete, Day 3 is next.")

    profiles = [
        (
            "Conflict profile: high energy + sad",
            {
                "genre": "pop",
                "mood": "sad",
                "energy": 0.95,
                "likes_acoustic": True,
            },
        ),
        (
            "Unknown mood fallback",
            {
                "genre": "lofi",
                "mood": "bittersweet",
                "energy": 0.6,
                "likes_acoustic": True,
            },
        ),
        (
            "Out-of-range energy",
            {
                "genre": "pop",
                "mood": "happy",
                "energy": 1.8,
                "likes_acoustic": False,
            },
        ),
    ]

    _print_section("Adversarial Profile Results")
    for profile_name, user_prefs in profiles:
        recommendations = recommend_songs(user_prefs, songs, k=3)
        _print_recommendation_table(profile_name, user_prefs, recommendations)

    _print_section("Agent 1 To Recommender Demo")
    agent_messages = [
        "Need hype songs for a workout run tonight",
        "Give me calm lofi study music",
    ]
    for message in agent_messages:
        user_prefs, mood_payload = _prefs_from_agent_message(message)
        recommendations = recommend_songs(user_prefs, songs, k=3)
        print(f"message: {message}")
        _print_agent_payload(mood_payload)
        _print_recommendation_table("Agent-derived profile", user_prefs, recommendations)


if __name__ == "__main__":
    main()
