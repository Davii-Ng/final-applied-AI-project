<h1 align="center">🎧 DJ Music Recommender</h1>

<p align="center">
  <em>Tell it your vibe. Watch four AI agents turn it into a setlist.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Tests-53%20passing-brightgreen?style=flat-square&logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/Agents-4-purple?style=flat-square" alt="Agents">
  <img src="https://img.shields.io/badge/Works%20Offline-yes-orange?style=flat-square" alt="Offline">
  <img src="https://img.shields.io/badge/Built%20for-CodePath%20AI110-red?style=flat-square" alt="CodePath">
</p>

---

## What It Does

You type `"hit the gym"` and a 4-agent pipeline:

1. **detects** your mood (intense, 0.59 confidence)
2. **builds** a listener profile (high-energy, danceable)
3. **retrieves + ranks** songs from a 40-track catalog using cosine similarity across 7 audio features
4. **writes** a DJ narration paragraph with Gemini

Every step is logged, scored, and explained — no black-box recommendations.

---

## Live Demo

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
  ─────────────────────────────────────────────────────────────────────────────────
  1    Midnight Protocol         Neon Echo             audio profile match (+7.21)
  2    Static Dreams             Volt Circuit          mood match: moody (+2.00)
  3    Chrome Horizon            Cyber Drift           audio profile match (+6.94)

  Sink into this moody synthwave ride — Midnight Protocol sets the tone with its
  pulsing bass and cold, cinematic textures, Static Dreams drifts you deeper into
  introspection, and Chrome Horizon closes the journey on the open road.
```

---

## Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                     USER VIBE INPUT                          ║
║         "time to hit the gym" / "a bit tired today"          ║
╚═══════════════════════════╤══════════════════════════════════╝
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 1 — Mood Detector                                  │
│  sentence-transformers (all-MiniLM-L6-v2)                 │
│  + keyword/phrase hybrid boost                            │
│  → detected_mood, confidence, energy_hint                 │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 2 — Profile Builder                                │
│  Rule-based genre / energy / constraint parser            │
│  Handles negation: "no edm", "avoid rap"                  │
│  → favorite_genre, target_energy, avoid_genres            │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 3 — Setlist Curator  [agentic state machine]       │
│                                                           │
│   plan → retrieve → check confidence                      │
│               ↑          │                                │
│               └── retry ←┘ (if conf < 0.5)               │
│                          │                                │
│                        rank → finalize                    │
│                                                           │
│  Retrieval: Gemini semantic + KB grounding                │
│             OR token-overlap fallback (no key needed)     │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│  AGENT 4 — DJ Narrator                                    │
│  Gemini writes a 2–3 sentence paragraph                   │
│  matching mood, genre, and setlist                        │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
╔══════════════════════════════════════════════════════════════╗
║   CLI: mood · energy · reasoning trace · setlist · narration ║
╚══════════════════════════════════════════════════════════════╝
```

### What makes it agentic

Agent 3 is a **self-correcting state machine** — not a single LLM call. It evaluates its own retrieval quality and retries with a broader search if confidence is low:

| Step | What happens |
|------|-------------|
| `plan` | Reads profile, logs catalog size + retry budget |
| `retrieve` | Calls Gemini semantic search (or token-overlap fallback) |
| `check_confidence` | If score < 0.5 and retries remain → retry |
| `retry` | Clears avoid-genres, doubles candidate pool |
| `rank` | Cosine similarity across 7 audio features |
| `finalize` | Assembles setlist with per-song explanations |

Every step is logged and shown in the CLI under **Reasoning Steps**.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Mood detection | `sentence-transformers` · `all-MiniLM-L6-v2` · keyword hybrid |
| Semantic retrieval | Google Gemini via `langchain-google-genai` |
| Ranking | Cosine similarity (NumPy) across 7 audio features |
| Narration | Gemini generative paragraph (`temperature=0.4`) |
| Grounding data | 40-song CSV catalog + 18-doc knowledge base (JSON) |
| Testing | `pytest` · 53 tests · smoke + unit + integration |
| Evaluation | Custom 12-case eval harness (`eval_harness.py`) |

---

## Quick Start

**1. Set up environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**2. Add your API key** *(optional — only needed for Gemini features)*

```bash
# .env file in project root
GOOGLE_API_KEY=your_key_here
```

> Without a key: Agent 1 and Agent 2 run fully locally, Agent 3 uses token-overlap retrieval, and Agent 4 uses a template paragraph. The core pipeline still works.

**3. Run it**

```bash
python -m src.cli
```

**4. Run tests**

```bash
python -m pytest -q .
```

**5. Run the evaluation harness**

```bash
python eval_harness.py --agentic   # 12 cases, all 9 moods
```

---

## Sample Interactions

### 🏋️ Workout mode
> `"Time to hit the gym"`

Mood → `intense` · Energy → high · Genre → pop  
Retriever pulls fast, high-danceability tracks with aggressive energy profile.

---

### 📚 Study session
> `"Give me calm lofi study music"`

Mood → `focused` · Energy → low · Genre → lofi  
Profile favors lower energy, high instrumentalness, acoustic-friendly songs.

---

### 🌧️ Moody night drive
> `"dark brooding night drive music"`

Mood → `moody` · Energy → medium · Genre → synthwave  
Narrator writes a full atmospheric paragraph matching the tone.

---

### 😴 Tired and winding down
> `"A bit tired today"`

Mood → `relaxed` · Energy → low  
Setlist shifts toward gentle, low-tempo tracks with high acousticness.

---

## Recommender Scoring

Each song is scored against a 7-dimensional user vector:

```
score = cosine_similarity(song_vec, user_vec) × 7.0
      + mood_match bonus          (up to +2.0)
      + genre_exact_match bonus   (+1.0)
      + mood_tag bonus            (+0.5)
      − diversity_penalties       (−2.0 repeat artist / −1.0 repeat genre)
```

Audio features in the vector:

```
[ energy · valence · tempo_norm · danceability · acousticness · instrumentalness · brightness ]
```

All features are normalized to `[0, 1]`. Tempo is mapped from `[60, 200 BPM] → [0, 1]`.

---

## Project Structure

```
├── src/
│   ├── cli.py                 ← interactive entry point
│   ├── orchestrator.py        ← 4-agent pipeline wiring + logging
│   ├── recommender.py         ← cosine scoring + greedy ranked selection
│   ├── retrieval.py           ← Gemini semantic retrieval + token-overlap fallback
│   ├── knowledge.py           ← KB loader + RAG context formatter
│   ├── models.py              ← Song and UserProfile dataclasses
│   └── agents/
│       ├── agent1_mood.py     ← mood detection (ST hybrid + local keyword)
│       ├── agent2_profile.py  ← profile builder (rule-based)
│       ├── agent3.py          ← agentic setlist curator (state machine)
│       └── agent4_narrator.py ← DJ narration (Gemini / template fallback)
│
├── data/
│   ├── songs.csv              ← 40-song catalog with audio features
│   └── knowledge_base.json   ← 18 KB docs (9 genres + 9 moods) for RAG
│
├── tests/                     ← 53 tests across all agents and modules
├── eval_harness.py            ← 12-case evaluation script
└── docs/
    ├── implementation_guide.md
    └── model_card.md
```

---

## Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| 4-agent pipeline instead of one LLM call | Each step is observable, testable, debuggable | More orchestration complexity |
| Deterministic cosine ranking | Predictable results, easy unit tests | Less flexible than generative ranking |
| Gemini retrieval + local fallback | Works with or without an API key | Fallback has lower semantic quality |
| Sentence-transformers + keyword hybrid | Handles short colloquial phrases ("hit the gym") that confuse pure embeddings | Slight complexity in scoring logic |
| Small curated dataset | Fast iteration, explainable outcomes | Limited diversity for niche tastes |

---

## Testing

```
53 passed in 8.15s
```

Coverage across:
- Agent 1: mood detection, confidence thresholds, ST + local backends
- Agent 2: profile parsing, negation, energy inference
- Agent 3: agentic state machine, retry logic, confidence gating
- Agent 4: narration with Gemini and template fallback
- Retrieval: Gemini path, token-overlap fallback, error surfacing
- Orchestrator + pipeline smoke tests

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Implementation Guide](docs/implementation_guide.md) | Full data contracts, agent signatures, retrieval pipeline, error handling |
| [Model Card & Reflection](docs/model_card.md) | Reliability methodology, behavioral findings, ethics, AI collaboration notes |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Agent 1 — mood detection via `all-MiniLM-L6-v2` |
| `langchain-google-genai` | Agent 3 retrieval + Agent 4 narration via Gemini |
| `langchain-core` | LangChain message primitives |
| `python-dotenv` | Loads API keys from `.env` |
| `numpy` | Cosine similarity math |
| `pytest` | Test runner |
