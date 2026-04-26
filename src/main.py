"""Demo runner for the rebuilt 4-agent music recommender pipeline."""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __package__ in {None, ""} and PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestrator import run_pipeline
from src.recommender import load_songs


def _print_section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def _print_pipeline_summary(result: dict) -> None:
    print(f"trace_id: {result.get('trace_id')}")
    print(f"agent1.mood: {result['agent1'].get('detected_mood')} (confidence={result['agent1'].get('confidence')})")

    profile = result["agent2"].get("profile", {})
    print(
        "agent2.profile: "
        f"genre={profile.get('favorite_genre')} | mood={profile.get('favorite_mood')} | "
        f"energy={profile.get('target_energy')} | likes_acoustic={profile.get('likes_acoustic')}"
    )

    setlist = result["agent3"].get("setlist", [])
    print(f"agent3.setlist_size: {len(setlist)}")
    print(f"agent4.intro: {result['agent4'].get('intro')}")

    for item in setlist:
        print(f"  #{item['rank']} {item['title']} - {item['artist']} (score={item['score']})")


def main() -> None:
    songs_csv = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(songs_csv)

    _print_section("Rebuilt Multi-Agent Demo")
    demo_messages = [
        "Need hype songs for a workout run tonight",
        "Give me calm lofi study music",
    ]

    for message in demo_messages:
        _print_section(f"Prompt: {message}")
        result = run_pipeline(user_message=message, songs=songs, k=3, agent1_backend="auto")
        _print_pipeline_summary(result)


if __name__ == "__main__":
    main()
