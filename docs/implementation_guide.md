# Implementation Guide

## 1. Purpose And Scope

This guide documents the current codebase behavior for the terminal-first music recommender and agent parsing modules. It is intended to let another engineer or coding agent implement, modify, or extend the project in one pass without guessing contracts.

Scope covered:
- CLI execution flow
- Recommendation scoring and ranking
- Agent 1 mood analysis contract
- Agent 2 profile parsing contract
- Agent 3 setlist curation contract
- Agent 4 narration contract
- Retrieval stage (src/retrieval.py)
- Orchestrator (src/orchestrator.py)
- Data model and CSV schema assumptions
- Test strategy and acceptance checks

Out of scope:
- UI/web frontend

## 2. System Context

The repository is a Python package with a deterministic ranking core and a four-agent pipeline:
- Recommender core: src/recommender.py
- Retrieval stage: src/retrieval.py
- Models: src/models.py
- Agent 1: src/agents/agent1_mood.py
- Agent 2: src/agents/agent2_profile.py
- Agent 3: src/agents/agent3_setlist.py
- Agent 4: src/agents/agent4_narrator.py
- Orchestrator: src/orchestrator.py
- CLI entries: src/main.py (demo runner) and src/cli.py (interactive)
- Test suites: tests/test_recommender.py, tests/test_agent1_mood.py, tests/test_agent2_profile.py, tests/test_agent3_setlist.py, tests/test_agent4_narrator.py, tests/test_orchestrator.py, tests/test_pipeline_smoke.py

The system reads a local CSV dataset and does not require network calls for normal ranking or profile parsing. Optional smoke testing uses Gemini when configured.

## 3. Architecture Overview

```mermaid
flowchart LR
  U[User Input] --> O[src/orchestrator.py]
  O --> A1[src/agents/agent1_mood.py]
  A1 --> A2[src/agents/agent2_profile.py]
  A2 --> A3[src/agents/agent3_setlist.py]
  A3 --> R[src/recommender.py recommend_songs]
  A3 --> A4[src/agents/agent4_narrator.py]
  D[data/songs.csv] --> R
  A4 --> C[CLI output]
```

Current state:
- Both src/main.py and src/cli.py call src/orchestrator.py.
- The orchestrator runs Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 with a shared trace_id.

Rebuilt module flow:

```mermaid
flowchart LR
  U[User Input] --> A1[analyze_mood]
  A1 --> A2[parse_profile]
  A2 --> A3[curate_setlist]
  A3 --> A4[narrate_setlist]
```

Secondary utility path:
- tests/test_connectivity_smoke.py performs optional Gemini connectivity smoke checks.

## 4. Data Contracts And Schemas

### 4.1 Song Dict Contract For Functional Recommender

Required keys consumed by recommend_songs:
- id: int
- title: str
- artist: str
- genre: str
- mood: str
- energy: float
- tempo_bpm: float
- valence: float
- danceability: float
- acousticness: float

Optional keys with defaults in load_songs:
- popularity: int (default 50)
- release_decade: int (default 2010)
- mood_tag: str (default mood or balanced)
- instrumentalness: float (default 0.2)
- vocal_presence: float (default 0.8)
- brightness: float (default 0.5)

### 4.2 User Preference Dict Contract

Keys used by functional scoring:
- genre: str
- mood: str
- energy: float
- likes_acoustic: bool

Invariants:
- Missing numeric fields are coerced with safe defaults.
- Non-numeric numeric fields fall back to default values.

### 4.3 Agent 1 Output Contract

Function: analyze_mood(user_message, optional_context=None, trace_id=None)

Output payload:
- schema_version: str
- trace_id: str
- detected_mood: str
- confidence: float in [0.0, 1.0]
- energy_hint: float in [0.0, 1.0] or null
- mood_candidates: list[str]
- notes: str

Validation and fallback:
- detected_mood must be an allowed mood label.
- If confidence < 0.55, detected_mood must be balanced.
- If no keywords match, mood_candidates becomes [balanced].

Example payload:

```json
{
  "schema_version": "1.0",
  "trace_id": "3b8f5f27-1b84-4dbf-8e7a-7f018cbd5f6f",
  "detected_mood": "happy",
  "confidence": 0.63,
  "energy_hint": 0.85,
  "mood_candidates": ["happy", "intense", "chill"],
  "notes": "keyword-based mood match: happy"
}
```

### 4.4 Agent 2 Input And Output Contract

Function: parse_profile(user_message, agent1_payload, optional_context=None, trace_id=None)

Agent 2 input fields:
- user_message: str
- agent1_payload: dict
  - detected_mood: str
  - confidence: float-like value
  - energy_hint: float-like value or null
  - trace_id: str optional
- optional_context: dict or null
  - favorite_genre: str optional
- trace_id: str optional

Output payload:
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
  - parser_mode: str (rules)
- request_summary: str

Validation and fallback:
- Invalid or unknown detected_mood is treated as balanced.
- If explicit mood is not present and Agent 1 confidence < 0.55, favorite_mood becomes balanced.
- Explicit numeric energy and energy keywords from user_message override energy_hint.
- If genre is not extractable from message, parser uses optional_context.favorite_genre if valid; otherwise default pop.
- All energy outputs are clamped to [0.0, 1.0].

Example payload:

```json
{
  "schema_version": "1.0",
  "trace_id": "trace-abc",
  "profile": {
    "favorite_genre": "hip-hop",
    "favorite_mood": "chill",
    "target_energy": 0.35,
    "likes_acoustic": true,
    "avoid_genres": ["edm"]
  },
  "constraints": {
    "missing_fields": [],
    "inferred_fields": ["likes_acoustic"],
    "low_confidence_mood": false,
    "disallowed_or_unknown_terms": [],
    "parser_mode": "rules"
  },
  "request_summary": "Prefers hip-hop with a chill vibe around energy 0.35."
}
```

## 5. Control Flow And Decision Points

```mermaid
sequenceDiagram
  participant User
  participant CLI as src/cli.py
  participant Orch as src/orchestrator.py
  participant Agent1 as analyze_mood
  participant Agent2 as parse_profile
  participant Ret as retrieve_candidates
  participant Agent3 as curate_setlist
  participant Agent4 as narrate_setlist

  User->>CLI: message text
  CLI->>Orch: run_pipeline(message, songs, k)
  Orch->>Agent1: analyze_mood(message)
  Agent1-->>Orch: mood payload + trace_id
  Orch->>Agent2: parse_profile(message, agent1_payload)
  Agent2-->>Orch: profile + constraints
  Orch->>Agent3: curate_setlist(agent2_payload, songs, k)
  Agent3->>Ret: retrieve_candidates(agent2_payload, songs)
  Ret-->>Agent3: candidate pool + retrieval debug
  Agent3-->>Orch: setlist + explanations + profile_echo
  Orch->>Agent4: narrate_setlist(agent3_payload)
  Agent4-->>Orch: intro + transitions + closing
  Orch-->>CLI: full pipeline result dict
  CLI-->>User: formatted table + narration
```

Decision points:
- Agent confidence gate:
  - confidence >= 0.55 uses top detected mood
  - confidence < 0.55 forces balanced fallback
- Energy hint precedence:
  - high-energy keywords override low-energy keywords
  - low-energy keywords used when no high-energy keyword is present
  - fallback to mood-based energy hint only when confidence is high enough
- Agent 2 profile precedence:
  - explicit mood in user message overrides Agent 1 mood
  - explicit energy in user message overrides Agent 1 energy_hint
  - optional_context genre is used only when user message has no valid genre
- Agent 3 retrieval gating:
  - avoid_genres filter applied before scoring
  - if retrieval returns no candidates, falls back to full catalog minus avoids
  - candidate pool size controlled by candidate_pool_size (default 20)
- Agent 4 persona:
  - concise style truncates transitions to max 3 entries
  - friendly style (default) returns all transitions

## 6. Error Handling And Fallback Behavior

Current state:
- Recommender parsing uses safe converters (_safe_float and _safe_int), avoiding crashes on malformed CSV values.
- Agent 1 handles empty input and ambiguous text by returning balanced fallback.
- Agent 2 handles invalid confidence, missing trace ids, unknown genre text, and invalid Agent 1 mood labels through rule-based fallback.
- Agent 3 returns an error payload (invalid_profile_payload) when agent2_payload.profile is not a dict; retrieval falls back to full catalog minus avoids if scoring produces no candidates.
- Agent 4 returns a canned fallback response when setlist is empty or not a list.
- Connectivity helper returns structured error payload instead of raising when API key/dependencies/network are missing.

Known fallback outcomes:
- missing GOOGLE_API_KEY -> {"ok": false, "error": "missing GOOGLE_API_KEY"}
- ambiguous mood text -> detected_mood = balanced
- missing genre signal in Agent 2 -> profile.favorite_genre = pop
- invalid energy hint in Agent 2 -> profile.target_energy = 0.55 (or explicit message energy if present)
- invalid Agent 3 profile input -> setlist = [], error = "invalid_profile_payload"
- empty setlist in Agent 4 -> canned intro and closing with safety_notes = ["empty_setlist_fallback"]

## 7. Setup And Run Commands

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Run interactive CLI:

```bash
python -m src.cli
```

Run demo runner (non-interactive):

```bash
python src/main.py
```

Optional script entrypoints (after editable install):

```bash
pip install -e .
dj-recommender        # demo runner
dj-recommender-cli    # interactive CLI
```

Run smoke tests only:

```bash
python -m pytest -q -m smoke
```

Smoke test behavior:
- If GOOGLE_API_KEY is missing, the smoke test is skipped.

## 8. Testing Strategy And Verification Commands

Current automated tests:
- tests/test_recommender.py
  - ranking order sanity
  - explanation string is non-empty
  - diversity penalty behavior
- tests/test_agent1_mood.py
  - payload schema validation
  - fallback confidence behavior
  - trace ID propagation
  - edge cases for empty input and energy-hint precedence
- tests/test_agent2_profile.py
  - profile schema validation
  - defaulting and fallback behavior for missing signals
  - explicit mood/energy precedence over Agent 1 hints
  - genre normalization and avoid_genres extraction
  - trace ID precedence and generation behavior
  - optional context genre handling
- tests/test_agent3_setlist.py
  - setlist schema validation
  - ranking and score output structure
  - retrieval debug metadata presence
  - invalid profile fallback behavior
- tests/test_agent4_narrator.py
  - narration schema validation
  - non-empty intro, transitions, and closing assertions
  - empty setlist fallback behavior
  - persona style (concise vs friendly) behavior
- tests/test_orchestrator.py
  - full pipeline schema validation
  - trace_id propagation across all four agents
  - end-to-end output structure
- tests/test_pipeline_smoke.py
  - real end-to-end run against local catalog

Run all tests:

```bash
python -m pytest -q .
```

Run focused suites:

```bash
python -m pytest -q tests/test_recommender.py
python -m pytest -q tests/test_agent1_mood.py
python -m pytest -q tests/test_agent2_profile.py
python -m pytest -q tests/test_agent3_setlist.py
python -m pytest -q tests/test_agent4_narrator.py
python -m pytest -q tests/test_orchestrator.py
```

## 9. Acceptance Criteria

A change is done when all are true:
- CLI runs without import errors from project root.
- src/main.py prints ranked recommendation tables.
- src/cli.py interactive loop runs end-to-end.
- Agent 1 output always includes all required fields.
- Agent 2 output always includes schema_version, trace_id, profile, constraints, and request_summary.
- Agent 3 output always includes schema_version, trace_id, setlist, explanations, profile_echo, and retrieval.
- Agent 4 output always includes schema_version, trace_id, intro, track_transitions, closing, and safety_notes.
- Agent 1 confidence fallback behavior remains intact at threshold 0.55.
- Agent 2 precedence and fallback behavior remains intact at threshold 0.55 for mood confidence handoff.
- trace_id propagates consistently from Agent 1 through all four agents.
- Full test suite passes.

## 10. Known Limitations And Open Questions

Known limitations:
- Catalog is small and static.
- Mood parser is keyword-based and English-centric.
- Agent 1, 2, and narration are all rule-based; no LLM calls are made in default (local/auto) mode.
- Agent 2 parser depends on explicit token patterns and will miss paraphrases.
- Retrieval stage (src/retrieval.py) is token-overlap based, not embedding-based.
- No persistent session state; each run_pipeline call is stateless.

Open questions:
- Should Gemini connectivity smoke checks remain test-only or integrate into CLI startup?
- Should retrieval move to embedding similarity for larger catalogs?
- What is the live-demo fallback if the optional Gemini backend fails?

## One-Shot Build Readiness Checklist

File-level implementation map:
- Edit src/agents/agent1_mood.py for mood behavior changes.
- Edit src/agents/agent2_profile.py for profile parsing changes.
- Edit src/agents/agent3_setlist.py for setlist curation changes.
- Edit src/agents/agent4_narrator.py for narration changes.
- Edit src/retrieval.py for retrieval stage changes.
- Edit src/recommender.py for scoring/weight changes.
- Edit src/orchestrator.py for pipeline wiring changes.
- Edit src/cli.py or src/main.py for CLI flow changes.
- Update corresponding tests/ files for regression coverage.

Public interfaces and signatures:
- src/agents/agent1_mood.py
  - analyze_mood(user_message: str, optional_context: dict | None = None, trace_id: str | None = None, backend: str = "local", model: str = "gemini-3-flash-preview", api_key: str | None = None) -> dict
  - class MoodAnalyst.analyze(...same args...) -> dict
- src/agents/agent2_profile.py
  - parse_profile(user_message: str, agent1_payload: dict, optional_context: dict | None = None, trace_id: str | None = None) -> dict
  - class ProfileParser.parse(...same args...) -> dict
- src/agents/agent3_setlist.py
  - curate_setlist(agent2_payload: dict, songs: list[dict], k: int = 5, candidate_pool_size: int = 20, trace_id: str | None = None) -> dict
  - class SetlistCurator.curate(...same args...) -> dict
- src/agents/agent4_narrator.py
  - narrate_setlist(agent3_payload: dict, persona: dict | None = None, trace_id: str | None = None) -> dict
  - class DJNarrator.narrate(...same args...) -> dict
- src/retrieval.py
  - retrieve_candidates(agent2_payload: dict, songs: list[dict], top_n: int) -> tuple[list[dict], dict]
- src/orchestrator.py
  - run_pipeline(user_message: str, songs: list[dict], k: int = 5, agent1_backend: str = "local", agent1_model: str = "gemini-3-flash-preview", agent1_api_key: str | None = None, optional_context: dict | None = None, persona: dict | None = None) -> dict
- src/recommender.py
  - load_songs(csv_path: str) -> list[dict]
  - recommend_songs(user_prefs: dict, songs: list[dict], k: int = 5) -> list[tuple[dict, float, list[str]]]
  - class Recommender.recommend(user: UserProfile, k: int = 5) -> list[Song]

Dependencies and environment assumptions:
- Python >= 3.11
- pandas
- pytest
- Optional for connectivity utility: langchain-google-genai, langchain-core, and python-dotenv

Step-by-step build order for a new feature:
1. Update data contracts in code and docs.
2. Implement Agent behavior changes in src/agents as needed.
3. Update retrieval or orchestrator wiring if the pipeline shape changes.
4. Add focused tests for new behavior.
5. Run targeted tests, then full suite.
6. Update README and this guide if interfaces changed.

Definition of done:
- Tests pass and cover new behavior.
- CLI output remains functional.
- Contracts and examples are updated.
- No unresolved TODOs in changed paths.

## Change Summary

- Updated guide to reflect full 4-agent pipeline (Agents 3 and 4 implemented and tested).
- Added retrieval stage (src/retrieval.py) contract and decision points.
- Updated orchestrator contract and CLI run instructions.
- Updated control flow sequence diagram to show current end-to-end flow.
- Updated acceptance criteria, testing, and known limitations to match current state.
