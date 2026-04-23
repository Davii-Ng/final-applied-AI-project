from typing import List, Dict, Tuple, Any
import csv
import math

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


def _score_song_data(song: Dict[str, Any], user_prefs: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    user_genre = str(user_prefs.get("genre", "")).lower()
    user_mood = str(user_prefs.get("mood", "")).lower()
    target_energy = _safe_float(user_prefs.get("energy", 0.5), 0.5)
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    song_genre = str(song.get("genre", "")).lower()
    song_mood = str(song.get("mood", "")).lower()
    song_mood_tag = str(song.get("mood_tag", song_mood)).lower()

    if song_genre == user_genre:
        score += 1.0
        reasons.append("genre match (+1.0)")

    if song_mood == user_mood:
        score += 5.0
        reasons.append("mood match (+5.0)")

    energy_score = _gaussian_similarity(_safe_float(song.get("energy", 0.0), 0.0), target_energy)
    score += energy_score * 3.5
    reasons.append(f"energy close to target (+{energy_score * 3.5:.2f})")

    tempo_target = 120.0 if target_energy >= 0.7 else 90.0
    tempo_score = _gaussian_similarity(_safe_float(song.get("tempo_bpm", 0.0), 0.0), tempo_target, sigma=20.0)
    score += tempo_score * 1.5
    reasons.append(f"tempo near preferred range (+{tempo_score * 1.5:.2f})")

    valence_target = 0.75 if user_mood == "happy" else 0.45 if user_mood in {"chill", "moody", "relaxed", "nostalgic", "sad"} else 0.6
    valence_score = _gaussian_similarity(_safe_float(song.get("valence", 0.0), 0.0), valence_target, sigma=0.18)
    score += valence_score * 1.5
    reasons.append(f"valence matches vibe (+{valence_score * 1.5:.2f})")

    danceability_target = 0.8 if target_energy >= 0.7 else 0.55
    danceability_score = _gaussian_similarity(_safe_float(song.get("danceability", 0.0), 0.0), danceability_target, sigma=0.18)
    score += danceability_score * 1.0
    reasons.append(f"danceability fit (+{danceability_score * 1.0:.2f})")

    if likes_acoustic:
        acoustic_score = _gaussian_similarity(_safe_float(song.get("acousticness", 0.0), 0.0), 0.8, sigma=0.2)
        score += acoustic_score * 1.0
        reasons.append(f"acoustic texture fit (+{acoustic_score * 1.0:.2f})")
    else:
        acoustic_score = _gaussian_similarity(_safe_float(song.get("acousticness", 0.0), 0.0), 0.2, sigma=0.2)
        score += acoustic_score * 0.5
        reasons.append(f"production style fit (+{acoustic_score * 0.5:.2f})")

    popularity_target = _preferred_popularity(user_genre, user_mood, target_energy, likes_acoustic)
    popularity_score = _gaussian_similarity(_safe_float(song.get("popularity", 50), 50.0), popularity_target, sigma=18.0)
    score += popularity_score * 1.2
    reasons.append(f"popularity near {int(round(popularity_target))} (+{popularity_score * 1.2:.2f})")

    decade_target = float(_preferred_decade(user_genre, user_mood, target_energy))
    decade_score = _gaussian_similarity(_safe_float(song.get("release_decade", 2010), 2010.0), decade_target, sigma=12.0)
    score += decade_score * 1.3
    reasons.append(f"release decade aligned with {int(decade_target)}s (+{decade_score * 1.3:.2f})")

    preferred_tags = _preferred_mood_tags(user_mood)
    if song_mood_tag in preferred_tags:
        score += 1.4
        reasons.append(f"mood tag fit ({song_mood_tag}) (+1.40)")

    instrumental_target = _preferred_instrumentalness(user_genre, user_mood, likes_acoustic)
    instrumental_score = _gaussian_similarity(_safe_float(song.get("instrumentalness", 0.0), 0.0), instrumental_target, sigma=0.18)
    score += instrumental_score * 1.1
    reasons.append(f"instrumentalness fit (+{instrumental_score * 1.1:.2f})")

    vocal_target = _preferred_vocal_presence(user_genre, user_mood, target_energy)
    vocal_score = _gaussian_similarity(_safe_float(song.get("vocal_presence", 0.0), 0.0), vocal_target, sigma=0.18)
    score += vocal_score * 1.0
    reasons.append(f"vocal presence fit (+{vocal_score * 1.0:.2f})")

    brightness_target = _preferred_brightness(user_genre, user_mood, target_energy)
    brightness_score = _gaussian_similarity(_safe_float(song.get("brightness", 0.0), 0.0), brightness_target, sigma=0.16)
    score += brightness_score * 1.0
    reasons.append(f"brightness fit (+{brightness_score * 1.0:.2f})")

    return score, reasons


def _diversity_penalty_values(
    artist: str,
    genre: str,
    chosen_pairs: List[Tuple[str, str]],
) -> Tuple[float, List[str]]:
    """Return a penalty for repeating artist or genre in top-ranked results."""
    penalty = 0.0
    reasons: List[str] = []

    if any(existing_artist == artist for existing_artist, _ in chosen_pairs):
        penalty += 2.0
        reasons.append("artist already in top results (-2.0)")

    if any(existing_genre == genre for _, existing_genre in chosen_pairs):
        penalty += 1.0
        reasons.append("genre already in top results (-1.0)")

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
        remaining = list(self.songs)
        chosen: List[Song] = []

        for _ in range(min(k, len(remaining))):
            best_song = None
            best_score = float("-inf")
            best_idx = -1

            for i, song in enumerate(remaining):
                base_score, _ = self._score_song(user, song)
                penalty, _ = self._diversity_penalty(song, chosen)
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

    remaining = list(scored)
    results: List[Tuple[Dict, float, List[str]]] = []
    chosen_songs: List[Dict] = []

    for _ in range(min(k, len(remaining))):
        best_idx = -1
        best_adjusted = float("-inf")
        best_reasons: List[str] = []

        for i, (song, base_score, reasons) in enumerate(remaining):
            chosen_pairs = [(s["artist"], s["genre"]) for s in chosen_songs]
            penalty, penalty_reasons = _diversity_penalty_values(song["artist"], song["genre"], chosen_pairs)
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
