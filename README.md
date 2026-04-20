# 🎵 Music Recommender Simulation

Builds playlist recommendations from a small handcrafted catalog using mood, genre, and audio-style features, then keeps top results diverse with artist/genre penalties.

## At a Glance

| What | Current State |
|---|---|
| Core engine | Functional API + OOP class |
| Data model | Split into `src/models.py` |
| Ranking logic | `src/recommender.py` |
| CLI output | Ranked ASCII table |
| Tests | `pytest` passing |

## Recent Improvements

- [x] Refactored dataclasses into `src/models.py` (`Song`, `UserProfile`)
- [x] Kept scoring/ranking logic centralized in `src/recommender.py`
- [x] Consolidated duplicate diversity penalty logic
- [x] Improved CLI readability (clear ranked table output)

## Quick Start

### 1) Install

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2) Run the App

```bash
python -m src.main
```

Alternative:

```bash
python src/main.py
```

### 3) Run Tests

```bash
python -m pytest
```

## How It Scores Songs

The engine combines exact-match boosts plus similarity-based scoring.

- Exact boosts: genre, mood, mood tags
- Gaussian similarity: energy, tempo, valence, danceability, acousticness, popularity, decade, instrumentalness, vocal presence, brightness
- Diversity guardrails after base ranking:
  - repeated artist: `-2.0`
  - repeated genre: `-1.0`

<details>
<summary><strong>Feature Inputs (click to expand)</strong></summary>

Each song has 16 fields:
- `genre`, `mood`, `mood_tag`
- `energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`
- `popularity`, `release_decade`
- `instrumentalness`, `vocal_presence`, `brightness`
- `artist`, `title`, `id`

User profile includes:
- favorite genre
- favorite mood
- target energy
- likes acoustic (boolean)

</details>

## Experiments So Far

- Adversarial profile: high energy + sad mood
- Unknown mood fallback profile (`bittersweet`)
- Out-of-range energy behavior
- Weight shift sensitivity (energy up, genre down)
- Diversity penalty behavior
- Mood alias sensitivity check (`upbeat` vs `happy`)

Observed:
- Ranking is sensitive to energy weight.
- Strong single signals can dominate mixed preferences.
- Diversity penalty reduces near-duplicate top picks.

## Project Structure

```text
src/
  __init__.py      # Package exports
  models.py        # Song and UserProfile dataclasses
  recommender.py   # Core scoring and ranking logic
  main.py          # CLI runner (ranked table output)
tests/
  test_recommender.py
data/
  songs.csv
assets/
  image.png
  image-1.png
model_card.md
reflection.md
plan.MD
```

## Assets

- ![Project image](assets/image.png)
- ![Project image 2](assets/image-1.png)

## Limitations

- Catalog is small (10 songs)
- Mood interpretation is still coarse
- Rule-based diversity is not yet personalized
- Exact matches can over-weight narrow preferences

## Documentation

- [Model Card](model_card.md)
- [Reflection Notes](reflection.md)
- [Project Plan](plan.MD)

