# Music Recommender Simulation

Terminal-first music recommender with two cooperating layers:
- Deterministic ranking in src/recommender.py
- Agent-style mood parsing in src/agents/agent1_mood.py

## What this project does

- Loads songs from data/songs.csv
- Scores songs against user preferences with weighted signals
- Applies diversity penalties to reduce repeated artist/genre picks
- Produces ranked output in a CLI table
- Demonstrates an Agent 1 mood-to-profile bridge in src/main.py

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
- src/main.py
  - Runs adversarial profiles and agent-derived profile demos

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

Fallback rule:
- If confidence is below 0.55, detected_mood is set to balanced.

Example:

```python
from src.agents.agent1_mood import analyze_mood

payload = analyze_mood("Need upbeat songs for a workout session")
print(payload)
```

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
tests/
  test_recommender.py
  test_agent1_mood.py
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

