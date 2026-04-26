from src.agents.agent1_mood import ALLOWED_MOODS, MoodAnalyst, analyze_mood


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeGeminiSuccess:
    def __init__(self, model: str, google_api_key: str, temperature: float, **kwargs):
        self.model = model
        self.google_api_key = google_api_key
        self.temperature = temperature

    def invoke(self, _messages):
        return type(
            "Response",
            (),
            {
                "content": (
                    '{"detected_mood":"happy","confidence":0.91,'
                    '"energy_hint":0.82,"mood_candidates":["happy","intense"],'
                    '"notes":"gemini test"}'
                )
            },
        )()


class _FakeGeminiBadJson:
    def __init__(self, model: str, google_api_key: str, temperature: float, **kwargs):
        self.model = model
        self.google_api_key = google_api_key
        self.temperature = temperature

    def invoke(self, _messages):
        return type("Response", (), {"content": "not-json"})()


def test_analyze_mood_returns_valid_schema_payload():
    payload = analyze_mood("I want upbeat happy songs for a party tonight")

    assert payload["schema_version"] == "1.0"
    assert isinstance(payload["trace_id"], str)
    assert payload["trace_id"].strip() != ""
    assert payload["detected_mood"] in ALLOWED_MOODS
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["mood_candidates"], list)
    assert len(payload["mood_candidates"]) >= 1
    assert isinstance(payload["notes"], str)
    assert payload["notes"].strip() != ""


def test_analyze_mood_falls_back_when_text_is_ambiguous():
    payload = analyze_mood("anything is fine i am not sure just play whatever")

    assert payload["detected_mood"] == "balanced"
    assert payload["confidence"] < 0.55
    assert payload["mood_candidates"] == ["balanced"]
    assert payload["notes"] == "low-confidence fallback to balanced"


def test_analyze_mood_honors_trace_id_and_agent_wrapper():
    analyst = MoodAnalyst()
    payload = analyst.analyze(
        user_message="I need intense workout tracks",
        trace_id="trace-123",
    )

    assert payload["trace_id"] == "trace-123"
    assert payload["detected_mood"] in {"intense", "balanced"}
    assert payload["energy_hint"] is None or 0.0 <= payload["energy_hint"] <= 1.0


def test_analyze_mood_handles_empty_or_none_message():
    payload = analyze_mood(None)  # type: ignore[arg-type]

    assert payload["detected_mood"] == "balanced"
    assert payload["confidence"] == 0.0
    assert payload["energy_hint"] is None
    assert payload["mood_candidates"] == ["balanced"]


def test_analyze_mood_high_energy_hint_wins_when_both_hint_types_present():
    payload = analyze_mood("happy workout calm chill")

    assert payload["energy_hint"] == 0.85


def test_analyze_mood_context_prior_mood_surfaces_in_candidates():
    payload = analyze_mood(
        user_message="upbeat happy",
        optional_context={"prior_mood": "nostalgic"},
    )

    assert payload["detected_mood"] == "happy"
    assert "nostalgic" in payload["mood_candidates"]


def test_analyze_mood_low_energy_keywords_force_low_energy_hint():
    payload = analyze_mood("I want a mellow relaxed lofi study session")

    assert payload["detected_mood"] in {"relaxed", "chill"}
    assert payload["energy_hint"] == 0.30


def test_analyze_mood_gemini_backend_returns_llm_payload_when_valid():
    payload = analyze_mood(
        user_message="Need upbeat workout tracks",
        backend="gemini",
        api_key="fake-key",
        llm_class=_FakeGeminiSuccess,
        message_class=_FakeMessage,
    )

    assert payload["detected_mood"] == "happy"
    assert payload["confidence"] == 0.91
    assert payload["energy_hint"] == 0.82
    assert payload["mood_candidates"] == ["happy", "intense"]
    assert payload["notes"] == "gemini test"


def test_analyze_mood_gemini_backend_falls_back_to_local_when_invalid_json():
    payload = analyze_mood(
        user_message="just play something",
        backend="gemini",
        api_key="fake-key",
        llm_class=_FakeGeminiBadJson,
        message_class=_FakeMessage,
    )

    assert payload["detected_mood"] in ALLOWED_MOODS
    assert "gemini fallback" in payload["notes"]
