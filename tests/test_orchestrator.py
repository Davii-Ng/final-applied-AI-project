from src.orchestrator import run_pipeline


def _songs() -> list[dict]:
    return [
        {
            "id": 1,
            "title": "Test One",
            "artist": "A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "valence": 0.9,
            "danceability": 0.8,
            "acousticness": 0.2,
            "popularity": 70,
            "release_decade": 2010,
            "mood_tag": "happy",
            "instrumentalness": 0.2,
            "vocal_presence": 0.8,
            "brightness": 0.7,
        }
    ]


def test_run_pipeline_returns_all_agent_payloads():
    result = run_pipeline(user_message="upbeat pop for workout", songs=_songs(), k=1, agent1_backend="local")

    assert isinstance(result["trace_id"], str)
    assert "agent1" in result
    assert "agent2" in result
    assert "agent3" in result
    assert "agent4" in result
    assert result["agent3"]["setlist"][0]["rank"] == 1
