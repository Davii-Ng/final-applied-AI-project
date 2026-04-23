from src.agents.agent2_profile import ProfileParser, parse_profile


def _agent1_payload(
    mood: str = "happy",
    confidence: float = 0.9,
    energy_hint: float | None = 0.7,
    trace_id: str = "trace-abc",
) -> dict:
    return {
        "schema_version": "1.0",
        "trace_id": trace_id,
        "detected_mood": mood,
        "confidence": confidence,
        "energy_hint": energy_hint,
        "mood_candidates": [mood],
    }


def test_parse_profile_returns_valid_schema():
    payload = parse_profile(
        user_message="I want upbeat pop songs for a workout",
        agent1_payload=_agent1_payload(),
    )

    assert payload["schema_version"] == "1.0"
    assert payload["trace_id"] == "trace-abc"
    assert isinstance(payload["request_summary"], str)

    profile = payload["profile"]
    assert isinstance(profile["favorite_genre"], str)
    assert isinstance(profile["favorite_mood"], str)
    assert isinstance(profile["target_energy"], float)
    assert 0.0 <= profile["target_energy"] <= 1.0
    assert isinstance(profile["likes_acoustic"], bool)
    assert isinstance(profile["avoid_genres"], list)

    constraints = payload["constraints"]
    assert isinstance(constraints["missing_fields"], list)
    assert isinstance(constraints["inferred_fields"], list)
    assert isinstance(constraints["low_confidence_mood"], bool)
    assert constraints["parser_mode"] == "rules"


def test_parse_profile_uses_defaults_for_missing_signals():
    payload = parse_profile(
        user_message="",
        agent1_payload=_agent1_payload(mood="balanced", confidence=0.0, energy_hint=None),
    )

    profile = payload["profile"]
    assert profile["favorite_genre"] == "pop"
    assert profile["favorite_mood"] == "balanced"
    assert profile["target_energy"] == 0.55
    assert profile["avoid_genres"] == []

    constraints = payload["constraints"]
    assert "favorite_genre" in constraints["missing_fields"]
    assert "target_energy" in constraints["missing_fields"]
    assert constraints["low_confidence_mood"] is True


def test_parse_profile_user_mood_overrides_agent1_when_explicit():
    payload = parse_profile(
        user_message="I need calm acoustic tracks",
        agent1_payload=_agent1_payload(mood="intense", confidence=0.95, energy_hint=0.9),
    )

    profile = payload["profile"]
    assert profile["favorite_mood"] == "relaxed"
    assert profile["likes_acoustic"] is True
    assert profile["target_energy"] == 0.3


def test_parse_profile_extracts_avoid_genres_and_normalizes():
    payload = parse_profile(
        user_message="Give me chill music, no rap and avoid edm",
        agent1_payload=_agent1_payload(mood="chill", confidence=0.9, energy_hint=0.35),
    )

    assert set(payload["profile"]["avoid_genres"]) == {"hip-hop", "edm"}


def test_parse_profile_clamps_numeric_energy_to_unit_interval():
    payload = parse_profile(
        user_message="energy 1.8 and upbeat vibes",
        agent1_payload=_agent1_payload(),
    )

    assert payload["profile"]["target_energy"] == 1.0


def test_profile_parser_wrapper_preserves_trace_id_override():
    parser = ProfileParser()
    payload = parser.parse(
        user_message="indie and nostalgic",
        agent1_payload=_agent1_payload(trace_id="orig-trace"),
        trace_id="override-trace",
    )

    assert payload["trace_id"] == "override-trace"


def test_parse_profile_uses_agent_mood_at_confidence_threshold():
    payload = parse_profile(
        user_message="Need something for lifting",
        agent1_payload=_agent1_payload(mood="intense", confidence=0.55, energy_hint=0.6),
    )

    assert payload["profile"]["favorite_mood"] == "intense"
    assert payload["constraints"]["low_confidence_mood"] is False
    assert "favorite_mood" in payload["constraints"]["inferred_fields"]


def test_parse_profile_invalid_confidence_falls_back_to_balanced():
    payload = parse_profile(
        user_message="",
        agent1_payload=_agent1_payload(mood="happy", confidence="high", energy_hint=0.4),
    )

    assert payload["profile"]["favorite_mood"] == "balanced"
    assert payload["constraints"]["low_confidence_mood"] is True
    assert "favorite_mood" in payload["constraints"]["inferred_fields"]


def test_parse_profile_invalid_detected_mood_defaults_to_balanced():
    payload = parse_profile(
        user_message="",
        agent1_payload=_agent1_payload(mood="mystery", confidence=0.9, energy_hint=0.4),
    )

    assert payload["profile"]["favorite_mood"] == "balanced"
    assert payload["constraints"]["low_confidence_mood"] is False
    assert "favorite_mood" in payload["constraints"]["inferred_fields"]


def test_parse_profile_uses_optional_context_genre_when_message_lacks_genre():
    payload = parse_profile(
        user_message="play something calm",
        agent1_payload=_agent1_payload(),
        optional_context={"favorite_genre": "hip hop"},
    )

    assert payload["profile"]["favorite_genre"] == "hip-hop"
    assert "favorite_genre" in payload["constraints"]["inferred_fields"]
    assert "favorite_genre" not in payload["constraints"]["missing_fields"]


def test_parse_profile_ignores_invalid_optional_context_genre():
    payload = parse_profile(
        user_message="play something calm",
        agent1_payload=_agent1_payload(),
        optional_context={"favorite_genre": "unknown genre"},
    )

    assert payload["profile"]["favorite_genre"] == "pop"
    assert "favorite_genre" in payload["constraints"]["missing_fields"]


def test_parse_profile_generates_trace_id_when_missing_or_blank():
    payload = parse_profile(
        user_message="indie please",
        agent1_payload=_agent1_payload(trace_id="   "),
        trace_id=" ",
    )

    assert isinstance(payload["trace_id"], str)
    assert payload["trace_id"].strip() != ""
    assert payload["trace_id"] != "trace-abc"


def test_parse_profile_clamps_energy_hint_when_no_explicit_energy():
    payload = parse_profile(
        user_message="nostalgic indie vibes",
        agent1_payload=_agent1_payload(energy_hint=2.2),
    )

    assert payload["profile"]["target_energy"] == 1.0
    assert "target_energy" in payload["constraints"]["inferred_fields"]


def test_parse_profile_invalid_energy_hint_uses_default_energy():
    payload = parse_profile(
        user_message="nostalgic indie vibes",
        agent1_payload=_agent1_payload(energy_hint="nope"),
    )

    assert payload["profile"]["target_energy"] == 0.55
    assert "target_energy" in payload["constraints"]["inferred_fields"]


def test_parse_profile_acoustic_and_non_acoustic_conflict_uses_mood_inference():
    payload = parse_profile(
        user_message="chill acoustic edm mix",
        agent1_payload=_agent1_payload(mood="happy", confidence=0.9, energy_hint=0.6),
    )

    assert payload["profile"]["favorite_mood"] == "chill"
    assert payload["profile"]["likes_acoustic"] is True


def test_parse_profile_supports_standalone_numeric_energy_signal():
    payload = parse_profile(
        user_message="let us go with 0.25 energy and relaxed tunes",
        agent1_payload=_agent1_payload(energy_hint=0.8),
    )

    assert payload["profile"]["target_energy"] == 0.25
