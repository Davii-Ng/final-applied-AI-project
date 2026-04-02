from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
import math

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _gaussian_similarity(value: float, target: float, sigma: float = 0.15) -> float:
        """Return a Gaussian similarity score for two numeric values."""
        return math.exp(-((value - target) ** 2) / (2 * sigma ** 2))

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Compute a recommendation score and explanation reasons for one song."""
        score = 0.0
        reasons: List[str] = []

        if song.genre == user.favorite_genre:
            score += 2.0
            reasons.append("genre match (+2.0)")

        if song.mood == user.favorite_mood:
            score += 5.0
            reasons.append("mood match (+5.0)")

        energy_score = self._gaussian_similarity(song.energy, user.target_energy)
        score += energy_score * 3.0
        reasons.append(f"energy close to target (+{energy_score * 3.0:.2f})")

        tempo_target = 120.0 if user.target_energy >= 0.7 else 90.0
        tempo_score = self._gaussian_similarity(song.tempo_bpm, tempo_target, sigma=20.0)
        score += tempo_score * 1.5
        reasons.append(f"tempo near preferred range (+{tempo_score * 1.5:.2f})")

        valence_target = 0.75 if user.favorite_mood == "happy" else 0.45 if user.favorite_mood in {"chill", "moody", "relaxed"} else 0.6
        valence_score = self._gaussian_similarity(song.valence, valence_target, sigma=0.18)
        score += valence_score * 1.5
        reasons.append(f"valence matches vibe (+{valence_score * 1.5:.2f})")

        danceability_score = self._gaussian_similarity(song.danceability, 0.8 if user.target_energy >= 0.7 else 0.55, sigma=0.18)
        score += danceability_score * 1.0
        reasons.append(f"danceability fit (+{danceability_score * 1.0:.2f})")

        if user.likes_acoustic:
            acoustic_score = self._gaussian_similarity(song.acousticness, 0.8, sigma=0.2)
            score += acoustic_score * 1.0
            reasons.append(f"acoustic texture fit (+{acoustic_score * 1.0:.2f})")
        else:
            acoustic_score = self._gaussian_similarity(song.acousticness, 0.2, sigma=0.2)
            score += acoustic_score * 0.5
            reasons.append(f"production style fit (+{acoustic_score * 0.5:.2f})")

        return score, reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs ranked by recommendation score."""
        scored_songs = sorted(
            ((song, self._score_song(user, song)[0]) for song in self.songs),
            key=lambda item: item[1],
            reverse=True,
        )
        return [song for song, _ in scored_songs[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a short human-readable explanation for a song score."""
        _, reasons = self._score_song(user, song)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and convert numeric fields for scoring."""
    songs: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )

    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, rank them, and return the top k recommendations."""
    def gaussian_similarity(value: float, target: float, sigma: float = 0.15) -> float:
        """Return a Gaussian similarity score for two numeric values."""
        return math.exp(-((value - target) ** 2) / (2 * sigma ** 2))

    recommendations: List[Tuple[Dict, float, List[str]]] = []

    for song in songs:
        score = 0.0
        reasons: List[str] = []

        if song["genre"] == user_prefs.get("genre"):
            score += 2.0
            reasons.append("genre match (+2.0)")

        if song["mood"] == user_prefs.get("mood"):
            score += 5.0
            reasons.append("mood match (+5.0)")

        target_energy = float(user_prefs.get("energy", 0.5))
        energy_score = gaussian_similarity(float(song["energy"]), target_energy)
        score += energy_score * 3.0
        reasons.append(f"energy close to target (+{energy_score * 3.0:.2f})")

        target_tempo = 120.0 if target_energy >= 0.7 else 90.0
        tempo_score = gaussian_similarity(float(song["tempo_bpm"]), target_tempo, sigma=20.0)
        score += tempo_score * 1.5
        reasons.append(f"tempo near preferred range (+{tempo_score * 1.5:.2f})")

        valence_target = 0.75 if user_prefs.get("mood") == "happy" else 0.45 if user_prefs.get("mood") in {"chill", "moody", "relaxed"} else 0.6
        valence_score = gaussian_similarity(float(song["valence"]), valence_target, sigma=0.18)
        score += valence_score * 1.5
        reasons.append(f"valence matches vibe (+{valence_score * 1.5:.2f})")

        danceability_score = gaussian_similarity(float(song["danceability"]), 0.8 if target_energy >= 0.7 else 0.55, sigma=0.18)
        score += danceability_score * 1.0
        reasons.append(f"danceability fit (+{danceability_score * 1.0:.2f})")

        if user_prefs.get("likes_acoustic"):
            acoustic_score = gaussian_similarity(float(song["acousticness"]), 0.8, sigma=0.2)
            score += acoustic_score * 1.0
            reasons.append(f"acoustic texture fit (+{acoustic_score * 1.0:.2f})")
        else:
            acoustic_score = gaussian_similarity(float(song["acousticness"]), 0.2, sigma=0.2)
            score += acoustic_score * 0.5
            reasons.append(f"production style fit (+{acoustic_score * 0.5:.2f})")

        recommendations.append((song, score, reasons))

    recommendations.sort(key=lambda item: item[1], reverse=True)
    return recommendations[:k]
