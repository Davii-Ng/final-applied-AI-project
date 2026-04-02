"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for song, score, reasons in recommendations:
        print(f"{song['title']} - Score: {score:.2f}")
        reason_text = ", ".join(reasons) if isinstance(reasons, list) else str(reasons)
        print(f"Because: {reason_text}")
        print()


if __name__ == "__main__":
    main()
