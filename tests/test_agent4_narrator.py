from src.agents.agent4_narrator import DJNarrator, narrate_setlist


def _agent3_payload() -> dict:
    return {
        "schema_version": "1.0",
        "trace_id": "trace-123",
        "setlist": [
            {"rank": 1, "title": "A", "artist": "X", "score": 12.3},
            {"rank": 2, "title": "B", "artist": "Y", "score": 11.9},
        ],
        "explanations": ["good mood match", "great energy fit"],
        "profile_echo": {"favorite_mood": "happy", "favorite_genre": "pop"},
    }


def test_narrate_setlist_returns_expected_fields():
    payload = narrate_setlist(_agent3_payload())

    assert payload["schema_version"] == "1.0"
    assert payload["trace_id"] == "trace-123"
    assert isinstance(payload["intro"], str) and payload["intro"].strip()
    assert isinstance(payload["track_transitions"], list) and len(payload["track_transitions"]) == 2
    assert isinstance(payload["closing"], str) and payload["closing"].strip()


def test_dj_narrator_wrapper_concise_persona_limits_transitions():
    narrator = DJNarrator()
    payload = narrator.narrate(_agent3_payload(), persona={"style": "concise"})

    assert len(payload["track_transitions"]) <= 2
