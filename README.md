# DJ Music Recommender

A terminal-first multi-agent music recommender built for CodePath AI110. Given a free-text "vibe" description, a 4-agent pipeline detects mood, builds a listener profile, curates a ranked setlist, and writes a DJ narration paragraph.

## Architecture

```
User vibe input
      |
  [Agent 1] Mood detection — sentence-transformers (all-MiniLM-L6-v2)
      |
  [Agent 2] Profile builder — rule-based genre/energy/constraint parser
      |
  [Agent 3] Setlist curator — agentic state machine
      |        plan -> retrieve -> check_confidence -> [retry] -> rank -> finalize
      |        Retrieval: Gemini semantic (with KB context) or token-overlap fallback
      |
  [Agent 4] DJ narrator — Gemini paragraph or template fallback
      |
  CLI output
```

### What makes it agentic

Agent 3 runs an observable state machine with a confidence-based retry loop:

1. **Plan** — reads profile, logs catalog size and retry budget
2. **Retrieve** — calls the retrieval tool (Gemini semantic or token-overlap)
3. **Check confidence** — if score < 0.5 and retries remain, go to Retry
4. **Retry** — clears `avoid_genres`, doubles the candidate pool, retrieves again
5. **Rank** — scores candidates with cosine similarity across 7 audio features
6. **Finalize** — assembles setlist with explanations

Every step is logged and displayed in the CLI under "Reasoning Steps".

### RAG / Knowledge Base

`data/knowledge_base.json` holds 18 documents (9 genres + 9 moods), each describing audio feature ranges. When Gemini retrieval is active, relevant KB documents are injected into the prompt as grounding context.

## Quick Start

**1. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API key (optional — Gemini features only)**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

or

```
GEMINI_API_KEY=your_key_here
```

Without an API key, Agent 3 falls back to token-overlap retrieval and Agent 4 uses a template paragraph. Agent 1 (sentence-transformers) and Agent 2 always run locally.

**4. Run the interactive CLI**

```bash
python -m src.cli
```

**5. Run the evaluation harness**

```bash
# Standard mode (single retrieve + rank)
python eval_harness.py

# Agentic mode (state machine with retry)
python eval_harness.py --agentic
```

**6. Run tests**

```bash
python -m pytest -q .
```

## CLI Output Example

```
DJ Recommender  [agentic mode]
----------------------------------------
How many songs? [default 3]: 3

Type your vibe. Enter 'quit' to exit.

Describe your vibe: dark brooding night drive music

  [Agent 1] Detecting mood (sentence-transformers)...
  [Agent 1] Done  — mood=moody (via sentence-transformers)
  [Agent 2] Building listener profile...
  [Agent 2] Done  — genre=synthwave
  [Agent 3] Curating setlist (agentic, retriever=gemini-semantic)...
  [Agent 3] Done  — 3 tracks (retriever=gemini-semantic)
  [Agent 4] Writing narration (gemini)...
  [Agent 4] Done  — narration ready

  Mood: moody        Energy: medium    Genre: synthwave

  Reasoning Steps:
  [plan]             analyze_profile_and_initialize
  [retrieve]         tool_call:retrieve_candidates (attempt 1)  conf=0.82  via=gemini-semantic  found=20
  [check_confidence] confidence=0.82 -> proceed to rank
  [rank]             tool_call:recommend_songs -> cosine_similarity_scoring  top=8.12
  [finalize]         setlist_assembled

  #    Title                     Artist                Why
  ─────────────────────────────────────────────────────────
  1    Midnight Protocol         Neon Echo             audio profile match (+7.21)
  2    Static Dreams             Volt Circuit          mood match: moody (+2.00)
  3    Chrome Horizon            Cyber Drift           audio profile match (+6.94)

  Sink into this moody synthwave ride with Midnight Protocol...
```

## Project Structure

```
src/
  cli.py              — interactive entry point
  main.py             — demo runner (batch mode)
  orchestrator.py     — runs the 4-agent pipeline with progress logging
  recommender.py      — cosine similarity scoring and greedy ranked selection
  retrieval.py        — Gemini semantic retrieval with token-overlap fallback
  knowledge.py        — knowledge base loader and context formatter
  models.py           — Song and UserProfile dataclasses
  agents/
    agent1_mood.py    — mood detection (sentence-transformers / Gemini / local)
    agent2_profile.py — listener profile builder (rule-based)
    agent3.py         — agentic setlist curator (state machine + simple mode)
    agent4_narrator.py — DJ narration (Gemini paragraph / template fallback)
data/
  songs.csv           — 40-song catalog with audio features
  knowledge_base.json — 18 KB documents (9 genres + 9 moods) for RAG context
tests/
  test_recommender.py
  test_retrieval.py
  test_agent1_mood.py
  test_agent2_profile.py
  test_agent3_setlist.py
  test_agent4_narrator.py
  test_orchestrator.py
  test_pipeline_smoke.py
  test_connectivity_smoke.py
eval_harness.py       — 12-case evaluation script with mood/genre/confidence metrics
docs/
  implementation_guide.md
```

## Agent Contracts

### Agent 1 — Mood Detection

**Backends:**
- `sentence_transformers` (default) — embeds user message and compares against mood reference sentences using `all-MiniLM-L6-v2`. ~13ms per call, no API key needed.
- `gemini` — uses Gemini to parse mood and return structured JSON with audio feature hints
- `local` — weighted keyword matching
- `auto` — tries Gemini, falls back to local

**Output fields:**

| Field | Type | Description |
|-------|------|-------------|
| `detected_mood` | str | One of: happy, chill, relaxed, moody, sad, intense, focused, nostalgic, balanced |
| `confidence` | float | Cosine similarity (ST) or model confidence (Gemini) |
| `energy_hint` | float | Derived energy level [0.0–1.0] |
| `mood_candidates` | list[str] | Top 3 mood candidates by score |
| `notes` | str | Backend used and score |

Fallback rule: if confidence < 0.38 (ST) or 0.55 (Gemini/local), mood is set to `balanced`.

### Agent 2 — Profile Builder

Converts Agent 1 output + user message into a recommender-ready profile. Always rule-based.

**Output — `profile` fields:**

| Field | Type |
|-------|------|
| `favorite_genre` | str |
| `favorite_mood` | str |
| `target_energy` | float [0–1] |
| `likes_acoustic` | bool |
| `avoid_genres` | list[str] |

Negation phrases like `"no edm"` or `"avoid rap"` populate `avoid_genres`.

### Agent 3 — Setlist Curator

Two modes via `agentic` parameter (default `True`):

**Agentic mode** — runs the full state machine (plan → retrieve → check → [retry] → rank → finalize). Returns `agentic_steps` list for observability.

**Simple mode** — single retrieve + rank pass. Used by `eval_harness.py` standard mode.

**Retrieval:**
- With `GOOGLE_API_KEY`: Gemini semantic retrieval — pre-filters to top 20 by token overlap, then sends to Gemini with KB context injected. Returns `retrieval_confidence` 0–1.
- Without key: token-overlap scoring with normalized confidence proxy.

**Output — `retrieval` debug fields:**

| Field | Description |
|-------|-------------|
| `retriever` | `"gemini-semantic"` or `"token-overlap"` |
| `candidates_after` | Pool size passed to ranker |
| `retrieval_confidence` | 0–1 confidence score |
| `kb_docs_injected` | Number of KB docs added to Gemini prompt |
| `filtered_avoid_genres` | Songs excluded by avoid list |

### Agent 4 — DJ Narrator

Writes a 2–3 sentence paragraph introducing the setlist.

- `gemini` backend: calls Gemini with `temperature=0.4`, `max_output_tokens=150`
- `local` backend: fills a template string

## Recommender Scoring

Scoring uses **cosine similarity** across 7 normalized audio features:

| Feature | Normalized range |
|---------|-----------------|
| Energy | [0, 1] |
| Valence | [0, 1] |
| Danceability | [0, 1] |
| Tempo BPM | [60, 200] → [0, 1] |
| Acousticness | [0, 1] |
| Instrumentalness | [0, 1] |
| Brightness | [0, 1] |

Audio cosine score is weighted ×7.0. On top of that:
- Mood match: up to +2.0
- Genre match: +1.0
- Mood tag match: +0.5

**Diversity penalties** applied during greedy selection:
- Repeated artist: −2.0
- Repeated genre: −1.0

## Evaluation

Run `python eval_harness.py --agentic` to evaluate 12 test cases covering all 9 moods against a local backend (no API key needed).

Metrics reported per case: mood match, genre match, retrieval confidence, avoid-genre violations.

Aggregate summary shows pass rate, mood/genre accuracy, and average confidence.

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Agent 1 mood detection (all-MiniLM-L6-v2) |
| `langchain-google-genai` | Gemini calls for Agent 3 retrieval and Agent 4 narration |
| `langchain-core` | LangChain message primitives |
| `python-dotenv` | Loads `GOOGLE_API_KEY` from `.env` |
| `pandas` | CSV loading |
| `numpy` | Cosine similarity math |
| `pytest` | Test runner |
