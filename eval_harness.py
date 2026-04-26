"""
Evaluation harness — runs the pipeline on predefined test cases using the
local backend (deterministic, no API key required) and prints a summary report.

Usage:
    python eval_harness.py
    python eval_harness.py --agentic      # use agentic Agent 3 workflow
"""

import os
import sys
import argparse
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.orchestrator import run_pipeline
from src.recommender import load_songs

# ---------------------------------------------------------------------------
# Test cases — all use local backend (no Gemini calls)
# ---------------------------------------------------------------------------

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "tc01",
        "description": "Gym workout -> intense",
        "user_message": "Need hype songs for a gym workout, pump it up",
        "expected_mood": "intense",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc02",
        "description": "Beast mode sprint -> intense",
        "user_message": "beast mode grind time sprint training aggressive",
        "expected_mood": "intense",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc03",
        "description": "Peaceful unwind -> relaxed",
        "user_message": "something peaceful and soothing to unwind and relax",
        "expected_mood": "relaxed",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc04",
        "description": "Late night brooding -> moody",
        "user_message": "dark brooding atmospheric music for a cloudy moody night",
        "expected_mood": "moody",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc05",
        "description": "Study session -> chill",
        "user_message": "chill lofi beats to study and relax with coffee",
        "expected_mood": "chill",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc06",
        "description": "Heartbreak -> sad",
        "user_message": "I feel sad and lonely and want to cry melancholy heartbreak songs",
        "expected_mood": "sad",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc07",
        "description": "Deep work deadline -> focused",
        "user_message": "I need to focus and concentrate on coding a big deadline productive",
        "expected_mood": "focused",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc08",
        "description": "Throwback memories -> nostalgic",
        "user_message": "throwback nostalgia retro classic memories from the old days",
        "expected_mood": "nostalgic",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc09",
        "description": "Party vibes -> happy",
        "user_message": "happy upbeat fun joyful party songs to celebrate",
        "expected_mood": "happy",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc10",
        "description": "Morning energy -> happy",
        "user_message": "cheerful euphoric songs to start the morning feeling excited and amazing",
        "expected_mood": "happy",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc11",
        "description": "Explicit avoid phrase -> intense, no edm",
        "user_message": "hype intense workout gym songs, no edm please avoid edm",
        "expected_mood": "intense",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
    {
        "id": "tc12",
        "description": "Vague request -> balanced fallback",
        "user_message": "just play me something nice",
        "expected_mood": "balanced",
        "expected_genre": None,
        "avoid_genres": [],
        "min_confidence": 0.0,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_test_case(
    tc: Dict[str, Any],
    songs: List[Dict],
    use_agentic: bool,
) -> Dict[str, Any]:
    try:
        result = run_pipeline(
            user_message=tc["user_message"],
            songs=songs,
            k=5,
            agent1_backend="local",
            agent4_backend="local",
            use_agentic=use_agentic,
        )
        return result
    except Exception as exc:
        return {"_error": str(exc)}


def _compute_metrics(tc: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    if "_error" in result:
        return {
            "id": tc["id"],
            "description": tc["description"],
            "mood_match": False,
            "genre_match": False,
            "retrieval_confidence": 0.0,
            "no_avoid_violations": False,
            "passed": False,
            "error": result["_error"],
        }

    detected_mood = result.get("agent1", {}).get("detected_mood", "")
    mood_match = (tc["expected_mood"] is None) or (detected_mood == tc["expected_mood"])

    detected_genre = result.get("agent2", {}).get("profile", {}).get("favorite_genre", "")
    genre_match = (tc["expected_genre"] is None) or (detected_genre == tc["expected_genre"])

    retrieval = result.get("agent3", {}).get("retrieval", {})
    confidence = float(retrieval.get("retrieval_confidence", 0.0))

    setlist = result.get("agent3", {}).get("setlist", [])
    setlist_titles = {s.get("title") for s in setlist}
    song_genre_map = {s.get("title"): s.get("genre") for s in result.get("_songs_ref", [])}

    violations = 0
    if tc["avoid_genres"] and song_genre_map:
        for title in setlist_titles:
            if song_genre_map.get(title, "") in tc["avoid_genres"]:
                violations += 1
    no_avoid_violations = violations == 0

    passed = (
        mood_match
        and genre_match
        and no_avoid_violations
        and confidence >= tc.get("min_confidence", 0.0)
    )

    return {
        "id": tc["id"],
        "description": tc["description"],
        "mood_match": mood_match,
        "genre_match": genre_match,
        "retrieval_confidence": confidence,
        "no_avoid_violations": no_avoid_violations,
        "passed": passed,
        "error": None,
    }


def _fmt_bool(val: bool) -> str:
    return "PASS" if val else "FAIL"


def _print_summary_table(metrics: List[Dict[str, Any]]) -> None:
    col_id    = 6
    col_desc  = 32
    col_mood  = 6
    col_genre = 7
    col_conf  = 11
    col_avoid = 9
    col_pass  = 7

    header = (
        f"{'ID':<{col_id}}"
        f"{'Description':<{col_desc}}"
        f"{'Mood':<{col_mood}}"
        f"{'Genre':<{col_genre}}"
        f"{'Confidence':<{col_conf}}"
        f"{'No-Avoid':<{col_avoid}}"
        f"{'Passed':<{col_pass}}"
    )
    divider = "-" * len(header)

    print("\n" + divider)
    print(header)
    print(divider)

    for m in metrics:
        error_flag = " !" if m["error"] else ""
        print(
            f"{m['id']:<{col_id}}"
            f"{(m['description'][:col_desc - 1]):<{col_desc}}"
            f"{_fmt_bool(m['mood_match']):<{col_mood}}"
            f"{_fmt_bool(m['genre_match']):<{col_genre}}"
            f"{m['retrieval_confidence']:<{col_conf}.4f}"
            f"{_fmt_bool(m['no_avoid_violations']):<{col_avoid}}"
            f"{('YES' if m['passed'] else 'NO') + error_flag:<{col_pass}}"
        )

    print(divider)


def _print_aggregate(metrics: List[Dict[str, Any]]) -> None:
    n = len(metrics)
    passed     = sum(1 for m in metrics if m["passed"])
    mood_hits  = sum(1 for m in metrics if m["mood_match"])
    genre_hits = sum(1 for m in metrics if m["genre_match"])
    avoid_ok   = sum(1 for m in metrics if m["no_avoid_violations"])
    avg_conf   = sum(m["retrieval_confidence"] for m in metrics) / n if n else 0.0
    errors     = sum(1 for m in metrics if m["error"])

    print(f"\nAggregate Results ({n} test cases)")
    print(f"  Pass rate            : {passed}/{n}  ({100 * passed / n:.1f}%)")
    print(f"  Mood match rate      : {mood_hits}/{n}  ({100 * mood_hits / n:.1f}%)")
    print(f"  Genre match rate     : {genre_hits}/{n}  ({100 * genre_hits / n:.1f}%)")
    print(f"  Avg retrieval conf.  : {avg_conf:.4f}")
    print(f"  Avoid violations ok  : {avoid_ok}/{n}  ({100 * avoid_ok / n:.1f}%)")
    if errors:
        print(f"  Pipeline errors      : {errors}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Music recommender evaluation harness")
    parser.add_argument("--agentic", action="store_true", help="Use agentic Agent 3 workflow")
    args = parser.parse_args()

    songs_path = os.path.join(PROJECT_ROOT, "data", "songs.csv")
    songs = load_songs(songs_path)
    song_genre_map = {s["title"]: s["genre"] for s in songs}

    mode = "agentic" if args.agentic else "standard"
    print(f"\nEvaluation Harness — {len(TEST_CASES)} test cases | backend: local | mode: {mode}")

    metrics: List[Dict[str, Any]] = []
    for tc in TEST_CASES:
        result = _run_test_case(tc, songs, use_agentic=args.agentic)
        result["_songs_ref"] = [{"title": t, "genre": g} for t, g in song_genre_map.items()]
        m = _compute_metrics(tc, result)
        metrics.append(m)
        status = "." if m["passed"] else "F"
        print(status, end="", flush=True)

    print()
    _print_summary_table(metrics)
    _print_aggregate(metrics)

    if args.agentic:
        print("Agentic step counts (per test case):")
        for i, tc in enumerate(TEST_CASES):
            result = _run_test_case(tc, songs, use_agentic=True)
            steps = result.get("agentic_steps", result.get("agent3", {}).get("agentic_steps", []))
            retry = result.get("agent3", {}).get("retry_triggered", False)
            print(f"  {tc['id']}: {len(steps)} steps | retry={retry}")


if __name__ == "__main__":
    main()
