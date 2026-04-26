# Agentic Music Recommender — Project Brief (v1)

**Status:** Draft for review
**Type:** Technical prototype, internal
**Owner:** [TBD]
**Audience for this doc:** Manager (for alignment), engineer/intern (for execution)

---

## TL;DR

A CLI tool that takes a natural-language vibe (e.g. *"lock in,"* *"sad drive home,"* *"gym but not aggressive"*) and returns a ranked, explained playlist drawn from a hand-labeled CSV of songs. Under the hood, a **multi-agent LLM system with RAG** handles intent parsing, retrieval, ranking, and narration. The CLI surfaces each agent's reasoning as it runs.

The point of this project is **not** to ship a consumer product. It is to demonstrate the team's capability with modern agentic AI architectures to internal leadership. Success = a compelling live demo.

---

## Why we're building this

The team wants a portfolio/learning project that shows applied competence with multi-agent systems and RAG. Music recommendation is the chosen domain because the problem is intuitive to non-technical stakeholders — anyone can evaluate "did this playlist match my vibe?" — while the underlying pipeline has enough complexity to showcase real architecture.

This is explicitly a **technical showcase**, not a product launch. Framing it otherwise sets leadership up to ask "where's the app?" and miss what's actually impressive.

---

## Primary user (for the prototype)

A **leadership stakeholder watching a demo**. Not an end consumer. They'll type a vibe, watch the system reason, and judge whether the approach is compelling and technically sound.

A secondary user is the **engineer/intern on the team**, who needs to understand the architecture well enough to extend it.

---

## What we're building — the wedge

The compelling story is not "better playlists." Leadership has seen a hundred recommendation demos. The wedge is **visible agentic deliberation**: watching four specialized agents coordinate to produce a result that a single LLM call couldn't.

Three things the demo must make obvious:

1. **Intent is parsed, not matched.** Weird vibes like *"aggressive but not angry, like before a big test"* get structured into real criteria, not keyword lookups.
2. **Retrieval is grounded.** RAG pulls from a known catalog; the system can't hallucinate songs that don't exist.
3. **Reasoning is legible.** The user sees each agent's contribution in real time — not just a final answer.

---

## Architecture

Four agents, orchestrated in sequence, with their reasoning streamed to the CLI:

| Agent | Input | Output | What it demonstrates |
|---|---|---|---|
| **Planner** | Raw user vibe | Structured taste profile (mood, energy, tempo range, genre hints, exclusions) | Intent decomposition |
| **Retriever** | Taste profile | Candidate songs from CSV via RAG over song labels/descriptions | Grounded retrieval |
| **Ranker** | Candidates + profile | Ranked list with per-song scores | Deterministic, testable scoring |
| **Narrator** | Ranked list + profile | Human-readable explanation for each pick | LLM synthesis |

The CLI prints each agent's output as it happens, so the demo audience sees the system think, not just its conclusion.

---

## Scope

### In scope for v1

- CLI interface, local execution only
- Hand-labeled CSV of a few hundred songs (labels: mood, energy, genre, tempo, context tags like "focus" / "gym" / "late night")
- 10–20 common vibes nailed well: *lock in, gym, sad, focus, drive, study, party, chill, sleep, heartbreak, hype, nostalgia*, etc.
- Four-agent pipeline as described above
- Streamed reasoning visible in the terminal
- Persistent local user profile file (so repeat runs can refine over time) — optional for v1 if time-constrained

### Explicitly out of scope

- Playback (no audio, no Spotify/Apple integration)
- GUI of any kind — no web, no mobile, no desktop app
- User accounts, authentication, multi-tenancy
- Large-scale catalog (Million Song Dataset, live API pulls)
- Latency or cost optimization
- Production deployment

If a stakeholder asks about any of the above, the answer is: *"That's a productization question. This prototype is about proving the architecture; turning it into a product is a separate, much larger project."*

---

## Success criteria

The project succeeds if, at demo time:

1. A stakeholder can type an arbitrary vibe and get a sensible playlist back within ~10–20 seconds.
2. The reasoning trace is legible enough that a non-technical viewer understands what each agent contributed.
3. The team can articulate why the multi-agent approach outperforms a single LLM call for this task — with at least one concrete example where a single-call baseline would have failed.
4. Leadership walks out saying *"that architecture is interesting"* rather than *"cool playlist."*

The project does **not** need to:
- Produce the best possible playlists in every case
- Handle adversarial inputs
- Scale beyond one user

---

## Open questions (things that still need a decision)

These are the things I'd push the manager to close before code is written:

1. **Does the project have a productization path after the demo, or is the demo itself the end state?** Affects how much we invest in code quality and modularity.
2. **Who hand-labels the CSV, and to what standard?** Label quality is the quiet bottleneck on retrieval quality. A bad CSV sinks the demo regardless of architecture.
3. **What's the fallback if agent orchestration is flaky during the live demo?** (Pre-recorded backup? Canned inputs that are known to work?)
4. **Is "multi-agent" the right framing, or are we actually building a pipeline of LLM calls?** These are different things; the distinction matters for how we pitch the work. Worth agreeing internally before presenting externally.

---

## What an intern could do on day one

Open the repo, read this doc, and start on any of:

- Schema the CSV: decide the label columns and hand-label ~50 seed songs
- Build the Planner agent in isolation: user vibe in, structured profile out, tested on ~10 example prompts
- Stand up a minimal RAG index over the CSV using any off-the-shelf vector store
- Write the CLI loop that takes input, calls a placeholder pipeline, and streams output

None of these blocks the others. The team can parallelize from day one.

---

## What this doc is not

It is not a product spec for a consumer app. If the project later grows into one, that's a new document, a new scope, and a new conversation with leadership about resourcing. Keep the scope small, ship the demo, then decide.
