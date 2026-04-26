from typing import List, Dict, Tuple, Any
import csv
import math

import numpy as np

from .models import Song, UserProfile


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _gaussian_similarity(value: float, target: float, sigma: float = 0.15) -> float:
    sigma = max(abs(float(sigma)), 1e-6)
    return math.exp(-((value - target) ** 2) / (2 * sigma ** 2))


# Fuzzy mood similarity — how much a song's mood contributes when it isn't an exact match.
# Symmetric: (a, b) covers both directions.
_MOOD_SIMILARITY: Dict[Tuple[str, str], float] = {
    ("happy",     "happy"):     1.0,
    ("happy",     "chill"):     0.3,
    ("happy",     "relaxed"):   0.2,
    ("happy",     "nostalgic"): 0.2,
    ("happy",     "intense"):   0.3,
    ("chill",     "chill"):     1.0,
    ("chill",     "relaxed"):   0.7,
    ("chill",     "nostalgic"): 0.4,
    ("chill",     "focused"):   0.4,
    ("chill",     "moody"):     0.3,
    ("relaxed",   "relaxed"):   1.0,
    ("relaxed",   "chill"):     0.7,
    ("relaxed",   "nostalgic"): 0.4,
    ("relaxed",   "sad"):       0.2,
    ("moody",     "moody"):     1.0,
    ("moody",     "sad"):       0.6,
    ("moody",     "nostalgic"): 0.5,
    ("moody",     "chill"):     0.3,
    ("sad",       "sad"):       1.0,
    ("sad",       "moody"):     0.6,
    ("sad",       "nostalgic"): 0.4,
    ("sad",       "relaxed"):   0.2,
    ("intense",   "intense"):   1.0,
    ("intense",   "focused"):   0.4,
    ("intense",   "happy"):     0.3,
    ("focused",   "focused"):   1.0,
    ("focused",   "intense"):   0.4,
    ("focused",   "chill"):     0.4,
    ("nostalgic", "nostalgic"): 1.0,
    ("nostalgic", "moody"):     0.5,
    ("nostalgic", "chill"):     0.4,
    ("nostalgic", "sad"):       0.4,
    ("nostalgic", "relaxed"):   0.4,
}


def _mood_similarity(user_mood: str, song_mood: str) -> float:
    if user_mood == "balanced":
        return 0.1  # neutral: mood bonus is noise, keep it near zero
    sim = _MOOD_SIMILARITY.get((user_mood, song_mood))
    if sim is None:
        sim = _MOOD_SIMILARITY.get((song_mood, user_mood), 0.0)
    return sim


def _preferred_mood_tags(user_mood: str) -> set[str]:
    mood_tag_map = {
        "happy": {"happy", "euphoric", "warm", "bright"},
        "chill": {"chill", "dreamy", "warm", "nostalgic"},
        "relaxed": {"relaxed", "warm", "dreamy", "nostalgic"},
        "moody": {"moody", "dreamy", "nostalgic"},
        "sad": {"sad", "dreamy", "nostalgic", "warm"},
        "intense": {"intense", "aggressive", "euphoric"},
        "focused": {"focused", "instrumental", "minimal", "clean"},
        "nostalgic": {"nostalgic", "warm", "dreamy"},
    }
    return mood_tag_map.get(user_mood, {user_mood})


def _preferred_decade(user_genre: str, user_mood: str, target_energy: float) -> int:
    if user_mood in {"nostalgic", "chill", "relaxed", "moody", "sad"}:
        return 1990
    if user_mood in {"focused"}:
        return 2000
    if user_mood in {"happy", "euphoric", "warm"} or user_genre in {"pop", "indie pop", "synthpop"}:
        return 2010
    if target_energy >= 0.85:
        return 2020
    return 2000


def _preferred_popularity(user_genre: str, user_mood: str, target_energy: float, likes_acoustic: bool) -> float:
    if user_genre in {"pop", "indie pop", "synthpop"} or user_mood in {"happy", "euphoric", "warm"}:
        return 78.0
    if user_mood in {"chill", "relaxed", "moody", "focused", "nostalgic", "sad"} or likes_acoustic:
        return 55.0
    if target_energy >= 0.85:
        return 70.0
    return 62.0


def _preferred_instrumentalness(user_genre: str, user_mood: str, likes_acoustic: bool) -> float:
    if user_mood in {"chill", "relaxed", "focused", "moody", "nostalgic", "sad"} or likes_acoustic:
        return 0.72
    if user_genre in {"pop", "indie pop", "synthpop"} or user_mood in {"happy", "euphoric"}:
        return 0.25
    return 0.45


def _preferred_vocal_presence(user_genre: str, user_mood: str, target_energy: float) -> float:
    if user_genre in {"pop", "indie pop", "synthpop"} or user_mood in {"happy", "euphoric", "warm"}:
        return 0.86
    if user_mood in {"chill", "relaxed", "focused", "moody", "nostalgic", "sad"}:
        return 0.44
    if target_energy >= 0.8:
        return 0.72
    return 0.58


def _preferred_brightness(user_genre: str, user_mood: str, target_energy: float) -> float:
    if user_genre in {"pop", "indie pop", "synthpop"} or user_mood in {"happy", "euphoric", "warm"}:
        return 0.78
    if user_mood in {"chill", "relaxed", "moody", "nostalgic", "sad"}:
        return 0.45
    if target_energy >= 0.8:
        return 0.66
    return 0.58


_TEMPO_MIN = 60.0
_TEMPO_MAX = 200.0


def _normalize_tempo(bpm: float) -> float:
    return max(0.0, min(1.0, (bpm - _TEMPO_MIN) / (_TEMPO_MAX - _TEMPO_MIN)))


def _build_song_vector(song: Dict[str, Any]) -> np.ndarray:
    return np.array([
        _safe_float(song.get("energy"), 0.5),
        _safe_float(song.get("valence"), 0.5),
        _normalize_tempo(_safe_float(song.get("tempo_bpm"), 120.0)),
        _safe_float(song.get("danceability"), 0.5),
        _safe_float(song.get("acousticness"), 0.2),
        _safe_float(song.get("instrumentalness"), 0.2),
        _safe_float(song.get("brightness"), 0.5),
    ], dtype=float)


def _build_user_vector(user_prefs: Dict[str, Any]) -> np.ndarray:
    user_genre = str(user_prefs.get("genre", "")).lower()
    user_mood = str(user_prefs.get("mood", "")).lower()
    target_energy = _safe_float(user_prefs.get("energy"), 0.5)
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    valence = user_prefs.get("target_valence")
    if valence is None:
        valence = 0.75 if user_mood == "happy" else 0.45 if user_mood in {"chill", "moody", "relaxed", "nostalgic", "sad"} else 0.6

    tempo_bpm = user_prefs.get("target_tempo_bpm")
    if tempo_bpm is None:
        tempo_bpm = 125.0 if target_energy >= 0.7 else 90.0

    danceability = user_prefs.get("target_danceability")
    if danceability is None:
        danceability = 0.8 if target_energy >= 0.7 else 0.55

    acousticness = user_prefs.get("target_acousticness")
    if acousticness is None:
        acousticness = 0.8 if likes_acoustic else 0.2

    instrumentalness = user_prefs.get("target_instrumentalness")
    if instrumentalness is None:
        instrumentalness = _preferred_instrumentalness(user_genre, user_mood, likes_acoustic)

    brightness = user_prefs.get("target_brightness")
    if brightness is None:
        brightness = _preferred_brightness(user_genre, user_mood, target_energy)

    return np.array([
        target_energy,
        float(valence),
        _normalize_tempo(float(tempo_bpm)),
        float(danceability),
        float(acousticness),
        float(instrumentalness),
        float(brightness),
    ], dtype=float)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)


def _score_song_data(song: Dict[str, Any], user_prefs: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    user_genre = str(user_prefs.get("genre", "")).lower()
    user_mood = str(user_prefs.get("mood", "")).lower()

    song_genre = str(song.get("genre", "")).lower()
    song_mood = str(song.get("mood", "")).lower()
    song_mood_tag = str(song.get("mood_tag", song_mood)).lower()

    # --- Continuous audio features via cosine similarity ---
    # All 7 features (energy, valence, tempo, danceability, acousticness,
    # instrumentalness, brightness) contribute equally — no feature dominates.
    song_vec = _build_song_vector(song)
    user_vec = _build_user_vector(user_prefs)
    audio_score = _cosine_similarity(song_vec, user_vec)
    score += audio_score * 7.0
    reasons.append(f"audio profile match (+{audio_score * 7.0:.2f})")

    # --- Categorical signals (can't live in a continuous vector) ---
    mood_sim = _mood_similarity(user_mood, song_mood)
    mood_contribution = mood_sim * 2.0
    if mood_contribution > 0:
        label = "mood match" if mood_sim == 1.0 else "mood similarity"
        score += mood_contribution
        reasons.append(f"{label} (+{mood_contribution:.2f})")

    if song_genre == user_genre:
        score += 1.0
        reasons.append("genre match (+1.0)")

    preferred_tags = _preferred_mood_tags(user_mood)
    if song_mood_tag in preferred_tags:
        score += 0.5
        reasons.append(f"mood tag fit ({song_mood_tag}) (+0.50)")

    return score, reasons


def _diversity_penalty_values(
    artist: str,
    genre: str,
    chosen_pairs: List[Tuple[str, str]],
    neutral: bool = False,
) -> Tuple[float, List[str]]:
    """Return a penalty for repeating artist or genre in top-ranked results.

    neutral=True (balanced mood) doubles the penalties to force variety when
    there is no strong signal to differentiate songs.
    """
    artist_penalty = 4.0 if neutral else 2.0
    genre_penalty  = 2.0 if neutral else 1.0
    penalty = 0.0
    reasons: List[str] = []

    if any(existing_artist == artist for existing_artist, _ in chosen_pairs):
        penalty += artist_penalty
        reasons.append(f"artist already in top results (-{artist_penalty:.1f})")

    if any(existing_genre == genre for _, existing_genre in chosen_pairs):
        penalty += genre_penalty
        reasons.append(f"genre already in top results (-{genre_penalty:.1f})")

    return penalty, reasons

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _diversity_penalty(song: Song, chosen_songs: List[Song]) -> Tuple[float, List[str]]:
        """Return a penalty for repeating an artist or genre already in the top results."""
        chosen_pairs = [(existing_song.artist, existing_song.genre) for existing_song in chosen_songs]
        return _diversity_penalty_values(song.artist, song.genre, chosen_pairs)

    @staticmethod
    def _gaussian_similarity(value: float, target: float, sigma: float = 0.15) -> float:
        """Return a Gaussian similarity score for two numeric values."""
        return _gaussian_similarity(value, target, sigma)

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        """Compute a recommendation score and explanation reasons for one song."""
        song_data = {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
            "popularity": song.popularity,
            "release_decade": song.release_decade,
            "mood_tag": song.mood_tag,
            "instrumentalness": song.instrumentalness,
            "vocal_presence": song.vocal_presence,
            "brightness": song.brightness,
        }
        user_data = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        return _score_song_data(song_data, user_data)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs ranked by recommendation score."""
        neutral = user.favorite_mood == "balanced"
        remaining = list(self.songs)
        chosen: List[Song] = []

        for _ in range(min(k, len(remaining))):
            best_song = None
            best_score = float("-inf")
            best_idx = -1

            for i, song in enumerate(remaining):
                base_score, _ = self._score_song(user, song)
                chosen_pairs = [(s.artist, s.genre) for s in chosen]
                penalty, _ = _diversity_penalty_values(song.artist, song.genre, chosen_pairs, neutral=neutral)
                if base_score - penalty > best_score:
                    best_score = base_score - penalty
                    best_song = song
                    best_idx = i

            if best_song is None:
                break

            chosen.append(best_song)
            remaining.pop(best_idx)

        return chosen

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
                    "id": _safe_int(row.get("id"), 0),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": _safe_float(row.get("energy"), 0.0),
                    "tempo_bpm": _safe_float(row.get("tempo_bpm"), 0.0),
                    "valence": _safe_float(row.get("valence"), 0.0),
                    "danceability": _safe_float(row.get("danceability"), 0.0),
                    "acousticness": _safe_float(row.get("acousticness"), 0.0),
                    "popularity": _safe_int(row.get("popularity"), 50),
                    "release_decade": _safe_int(row.get("release_decade"), 2010),
                    "mood_tag": row.get("mood_tag", row.get("mood", "balanced")),
                    "instrumentalness": _safe_float(row.get("instrumentalness"), 0.2),
                    "vocal_presence": _safe_float(row.get("vocal_presence"), 0.8),
                    "brightness": _safe_float(row.get("brightness"), 0.5),
                }
            )

    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score all songs, rank them, and return the top k recommendations."""
    scored: List[Tuple[Dict, float, List[str]]] = []
    for song in songs:
        base_score, reasons = _score_song_data(song, user_prefs)
        scored.append((song, base_score, reasons))

    neutral = str(user_prefs.get("mood", "")).lower() == "balanced"
    remaining = list(scored)
    results: List[Tuple[Dict, float, List[str]]] = []
    chosen_songs: List[Dict] = []

    for _ in range(min(k, len(remaining))):
        best_idx = -1
        best_adjusted = float("-inf")
        best_reasons: List[str] = []

        for i, (song, base_score, reasons) in enumerate(remaining):
            chosen_pairs = [(s["artist"], s["genre"]) for s in chosen_songs]
            penalty, penalty_reasons = _diversity_penalty_values(song["artist"], song["genre"], chosen_pairs, neutral=neutral)
            adjusted = base_score - penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i
                best_reasons = list(reasons) + penalty_reasons

        if best_idx == -1:
            break

        best_song, _, _ = remaining.pop(best_idx)
        results.append((best_song, best_adjusted, best_reasons))
        chosen_songs.append(best_song)

    return results
