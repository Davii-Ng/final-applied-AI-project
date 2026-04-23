# Music Recommender Simulation

Terminal-first music recommender with deterministic ranking and agent-style parsing:
- Deterministic ranking in src/recommender.py
- Agent-style mood parsing in src/agents/agent1_mood.py
- Agent-style profile parsing in src/agents/agent2_profile.py

## What this project does

- Loads songs from data/songs.csv
- Scores songs against user preferences with weighted signals
- Applies diversity penalties to reduce repeated artist/genre picks
- Produces ranked output in a CLI table
- Demonstrates an Agent 1 mood-to-profile bridge in src/main.py
- Includes a standalone Agent 2 parser contract validated by tests

## Quick Start

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the CLI.

```bash
python -m src.main
```

Alternative:

```bash
python src/main.py
```

Optional editable install and script entrypoint:

```bash
pip install -e .
dj-recommender
```

4. Run tests.

```bash
python -m pytest -q .
```

Run only smoke tests (real external connectivity):

```bash
python -m pytest -q -m smoke
```

Note:
- The smoke test requires GOOGLE_API_KEY.
- If the key is not set, the smoke test is skipped.

## Core Modules

- src/models.py
  - Song dataclass with audio and metadata fields
  - UserProfile dataclass for structured taste inputs
- src/recommender.py
  - Functional scoring and ranking pipeline
  - OOP Recommender class used by tests
- src/agents/agent1_mood.py
  - Converts raw user message into normalized mood payload
  - Enforces confidence-based fallback to balanced
  - Includes a short notes field describing the match or fallback
- src/agents/agent2_profile.py
  - Converts user message and Agent 1 payload into recommender-ready profile data
  - Normalizes genres, mood aliases, and energy inputs
  - Tracks inferred and missing fields through a constraints payload
- src/main.py
  - Runs adversarial profiles and Agent 1-derived profile demos

## Agent 1 Contract

Input:
- user_message: str
- optional_context: dict or None
- trace_id: str or None

Output JSON fields:
- schema_version: str
- trace_id: str
- detected_mood: str
- confidence: float in [0.0, 1.0]
- energy_hint: float in [0.0, 1.0] or None
- mood_candidates: list[str]
- notes: str

Fallback rule:
- If confidence is below 0.55, detected_mood is set to balanced.
- If the mood falls back, notes explains that the result was low-confidence.

Example:

```python
from src.agents.agent1_mood import analyze_mood

payload = analyze_mood("Need upbeat songs for a workout session")
print(payload)
```

## Agent 2 Contract

Input:
- user_message: str
- agent1_payload: dict
- optional_context: dict or None
- trace_id: str or None

Output JSON fields:
- schema_version: str
- trace_id: str
- profile: dict
  - favorite_genre: str
  - favorite_mood: str
  - target_energy: float in [0.0, 1.0]
  - likes_acoustic: bool
  - avoid_genres: list[str]
- constraints: dict
  - missing_fields: list[str]
  - inferred_fields: list[str]
  - low_confidence_mood: bool
  - disallowed_or_unknown_terms: list
  - parser_mode: str
- request_summary: str

Behavior highlights:
- Explicit mood keywords in user_message override Agent 1 mood.
- If Agent 1 confidence is below 0.55 and no explicit mood is present, mood falls back to balanced.
- Explicit energy values and high/low energy keywords override Agent 1 energy_hint.
- Unknown genre text falls back to context genre, then default pop.
- avoid_genres is extracted from negation phrases such as "no rap" and "avoid edm".

## Recommender Behavior Summary

- Strong boosts for exact mood and genre matches
- Gaussian similarity for energy, tempo, valence, danceability, acousticness, popularity, decade, instrumentalness, vocal presence, and brightness
- Diversity penalties applied after initial ranking:
  - repeated artist: -2.0
  - repeated genre: -1.0

## Project Structure

```text
src/
  __init__.py
  main.py
  models.py
  recommender.py
  agents/
    __init__.py
    agent1_mood.py
    agent2_profile.py
tests/
  test_recommender.py
  test_agent1_mood.py
  test_agent2_profile.py
  test_connectivity_smoke.py
data/
  songs.csv
assets/
  image.png
  image-1.png
pyproject.toml
requirements.txt
model_card.md
reflection.md
plan.MD
```

## Documentation

- model_card.md
- reflection.md
- plan.MD
- docs/implementation_guide.md

