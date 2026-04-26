# Reflection

## 1. Reliability and Evaluation: How I Tested and Improved the AI

I treated reliability as "can this system give consistent, explainable recommendations across different user inputs," not just "does it run once." I used four kinds of checks in this project:

### A. Automated tests
- I used unit tests for key modules (`recommender`, `retrieval`, `agent1_mood`, `agent2_profile`, `agent3_setlist`, `agent4_narrator`, `orchestrator`).
- I also used smoke tests for pipeline and connectivity behavior.
- Result summary from the project documentation: baseline test/eval expectations were fully met in normal runs (including the 12-case evaluation harness).

### B. Confidence scoring
- Agent 1 returns a confidence score for mood detection. The sentence-transformers backend uses a threshold of 0.20 (cosine similarity); local keyword backend uses 0.55.
- Agent 3 retrieval returns `retrieval_confidence` and uses a retry loop when confidence is low.
- In fallback token-overlap mode, average confidence was around `0.31` in evaluation; this helped me understand where retrieval is weaker and when retry/fallback behavior matters.
- When Gemini retrieval fails, the error is now surfaced explicitly in the retrieval debug dict (`gemini_error`) and printed as a `[warn]` line in the CLI, so failures are no longer silent.

### C. Logging and error handling
- The orchestrator prints per-agent progress and exposes step-by-step reasoning in agentic mode.
- Agent 3 and Agent 4 include fallback paths when Gemini is unavailable.
- Gemini retrieval failures are no longer silently swallowed — the error reason is captured and surfaced to the CLI so the cause of any token-overlap fallback is visible.
- Error cases are handled explicitly (for example: invalid profile payload, empty setlist, or missing API key), which prevents silent failures.

### D. Human evaluation
- I manually reviewed outputs for whether songs actually matched vibe intent (not just score values).
- I compared conflict-style prompts (mixed signals) versus clear prompts to see if the recommender behaved logically.
- This manual check was important because a numeric score can still produce a playlist that feels wrong to a real listener.

### Short reliability summary (rubric style)
- Automated testing and eval checks passed in baseline runs, including 12/12 evaluation cases in the documented setup.
- Confidence was strongest with clear mood/genre signals and weaker in token-overlap fallback mode (avg retrieval confidence ~0.31).
- Reliability improved after adding/using confidence thresholds, retry logic, and explicit fallbacks.

## 2. Behavioral Findings From Profile Comparisons

### Conflict profile vs. unknown mood profile
The conflict profile (very high energy + sad mood) often surfaced tracks like Gym Hero and Sunrise City because the scorer strongly rewards energy and mood matching. The unknown mood profile shifted toward lofi songs (for example Midnight Coding, Focus Flow, Library Rain) because the system used fallback behavior when mood was unclear. In simple terms, the first prompt sends mixed signals, while the second behaves like a broad calm-listening request.

### Conflict profile vs. out-of-range energy profile
The out-of-range energy profile moved brighter songs up because an unrealistic energy target weakens that feature's usefulness, so mood/genre signals dominate more. The conflict profile still ranked Gym Hero highly, but also allowed tracks like Storm Runner to rise due to tension between high-energy preference and sad mood. This shows how weight balance can unintentionally favor a few repeat songs.

### Unknown mood profile vs. out-of-range energy profile
The unknown mood profile stayed in chill/lofi territory because "bittersweet" was not recognized and fallback mood behavior kicked in. The out-of-range energy profile leaned toward happy pop patterns even with a broken energy value because genre/mood bonuses still contributed strongly. So one case looked "calm but undefined," while the other looked "happy pop with invalid energy input."

## 3. Reflection and Ethics

### What are the limitations or biases in this system?
- The catalog is small, so recommendations can become repetitive.
- The scoring design can over-reward strong mood/energy matches and under-serve mixed or niche tastes.
- Unknown mood words (for example, "bittersweet") are mapped to fallback behavior, which can flatten emotional nuance.
- The sentence-transformers hybrid approach works well for common phrases but can misclassify very short or unusual inputs where neither the embedding nor the keyword map has signal (fallback to `balanced` is the safe default in those cases).

### Could this AI be misused, and how would I prevent that?
- Misuse risk: presenting recommendations as if they were objective truth about a user's emotional state.
- Misuse risk: over-trusting results in sensitive contexts (mental health, crisis, etc.) where this tool is not appropriate.
- Mitigation: keep clear scope messaging in docs/CLI (entertainment/demo use), expose confidence and reasoning, and use conservative fallback behavior instead of pretending certainty.

### What surprised me while testing reliability?
What surprised me most was how much output ranking changed from small shifts in scoring weights or confidence handling. Even when the code looked stable, tiny tuning changes could reorder the top songs a lot. That made me value regression tests and repeatable eval cases more than I expected.

A concrete example: the original sentence-transformers confidence threshold (0.38) was set too high for MiniLM-L6-v2's cosine similarity range, so every input returned `balanced` regardless of actual mood. Inputs like "hit the gym" (clearly intense) and "I'm mad now" (clearly moody) were both collapsed to the same default. Lowering the threshold to 0.20 and adding a small keyword/phrase hybrid boost fixed detection without breaking any tests — but the root cause only became visible through manual test runs, not automated tests alone.

### Collaboration with AI during this project
- Helpful AI suggestion: AI helped me structure the multi-agent pipeline into separable responsibilities (mood detection, profile parsing, retrieval/ranking, narration), which made testing and debugging much easier. AI also diagnosed and fixed several bugs: a confidence threshold that was too high for the embedding model's similarity range, a silent exception swallow that hid Gemini retrieval failures, a `parsed` variable referenced before assignment in JSON parsing, and a narrator token cap that cut off sentences mid-paragraph.
- Flawed AI suggestion: AI once suggested wording/logic that sounded plausible but did not match actual project behavior (for example, overconfident assumptions about output quality in edge cases). I had to verify against tests and real runs, then correct the documentation/logic.

Main takeaway: AI accelerated drafting and iteration, but reliability came from verification, not from accepting AI-generated output at face value.
