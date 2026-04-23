# Music Recommender Simulation

Terminal-first multi-agent DJ recommender with a rebuilt 4-agent pipeline:
- Agent 1 mood analysis in src/agents/agent1_mood.py
- Agent 2 profile parsing in src/agents/agent2_profile.py
- Agent 3 setlist curation in src/agents/agent3_setlist.py
- Agent 4 narration in src/agents/agent4_narrator.py
- Unified orchestration in src/orchestrator.py

## What this project does

- Loads songs from data/songs.csv
- Runs a full 4-agent flow from user vibe to DJ narrative
- Uses deterministic ranking with weighted signals and diversity penalties
- Produces ranked output with reasons and narration text
- Supports interactive input in src/cli.py

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
python -m src.cli
```

Alternative:

```bash
python src/main.py
```

Optional editable install and script entrypoint:

```bash
pip install -e .
dj-recommender
dj-recommender-cli
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
- src/retrieval.py
  - Lightweight RAG-lite retrieval stage used by Agent 3
  - Filters songs by avoid_genres, scores by token overlap with profile, returns a candidate pool with debug metadata
- src/agents/agent1_mood.py
  - Converts raw user message into normalized mood payload
  - Enforces confidence-based fallback to balanced
  - Includes a short notes field describing the match or fallback
  - Supports local (rule-based), gemini, and auto backends
- src/agents/agent2_profile.py
  - Converts user message and Agent 1 payload into recommender-ready profile data
  - Normalizes genres, mood aliases, and energy inputs
  - Tracks inferred and missing fields through a constraints payload
- src/agents/agent3_setlist.py
  - Converts Agent 2 profile payload into ranked setlist output
  - Runs retrieval stage first, then deterministic scoring
  - Returns schema version, trace id, setlist, explanations, profile echo, and retrieval debug
- src/agents/agent4_narrator.py
  - Converts ranked setlist into intro, transitions, and closing narration
  - Supports persona dict with a style field (friendly or concise)
- src/orchestrator.py
  - Runs Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 with shared trace_id
  - Exposes run_pipeline with configurable backend and persona
- src/cli.py
  - Interactive runtime for the rebuilt pipeline
  - Prompts for k, backend, and output mode before the loop

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

## Agent 3 Contract

Input:
- agent2_payload: dict
- songs: list[dict]
- k: int
- candidate_pool_size: int (default 20)
- trace_id: str or None

Output JSON fields:
- schema_version: str
- trace_id: str
- setlist: list[dict]
  - rank: int
  - title: str
  - artist: str
  - score: float
- explanations: list[str]
- profile_echo: dict
- retrieval: dict (debug metadata from retrieval stage)
  - retriever: str
  - query_tokens: list[str]
  - candidates_before: int
  - candidates_after: int
  - filtered_avoid_genres: int
  - retrieval_fallback: bool

Behavior:
- Runs src/retrieval.py retrieve_candidates first to build a candidate pool
- Applies avoid_genres filter before scoring
- Falls back to full catalog minus avoids if retrieval returns no candidates

## Agent 4 Contract

Input:
- agent3_payload: dict
- persona: dict or None
- trace_id: str or None

Output JSON fields:
- schema_version: str
- trace_id: str
- intro: str
- track_transitions: list[str]
- closing: str
- safety_notes: list[str]

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
  cli.py
  models.py
  recommender.py
  retrieval.py
  orchestrator.py
  agents/
    __init__.py
    agent1_mood.py
    agent2_profile.py
    agent3_setlist.py
    agent4_narrator.py
tests/
  test_recommender.py
  test_agent1_mood.py
  test_agent2_profile.py
  test_agent3_setlist.py
  test_agent4_narrator.py
  test_orchestrator.py
  test_pipeline_smoke.py
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

