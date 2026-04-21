from src.agent1_mood import ALLOWED_MOODS, MoodAnalyst, analyze_mood


def test_analyze_mood_returns_valid_schema_payload():
    payload = analyze_mood("I want upbeat happy songs for a party tonight")

    assert payload["schema_version"] == "1.0"
    assert isinstance(payload["trace_id"], str)
    assert payload["trace_id"].strip() != ""
    assert payload["detected_mood"] in ALLOWED_MOODS
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["mood_candidates"], list)
    assert len(payload["mood_candidates"]) >= 1


def test_analyze_mood_falls_back_when_text_is_ambiguous():
    payload = analyze_mood("anything is fine i am not sure just play whatever")

    assert payload["detected_mood"] == "balanced"
    assert payload["confidence"] < 0.55
    assert payload["mood_candidates"] == ["balanced"]


def test_analyze_mood_honors_trace_id_and_agent_wrapper():
    analyst = MoodAnalyst()
    payload = analyst.analyze(
        user_message="I need intense workout tracks",
        trace_id="trace-123",
    )

    assert payload["trace_id"] == "trace-123"
    assert payload["detected_mood"] in {"intense", "balanced"}
    assert payload["energy_hint"] is None or 0.0 <= payload["energy_hint"] <= 1.0
