from src.agents.agent3 import SetlistCurator, curate_setlist


def _agent2_payload() -> dict:
    return {
        "schema_version": "1.0",
        "trace_id": "trace-xyz",
        "profile": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.8,
            "likes_acoustic": False,
            "avoid_genres": [],
        },
        "constraints": {},
        "request_summary": "happy pop high energy",
    }


def _songs() -> list[dict]:
    return [
        {
            "id": 1,
            "title": "Track One",
            "artist": "Artist A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.82,
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
        },
        {
            "id": 2,
            "title": "Track Two",
            "artist": "Artist B",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.3,
            "tempo_bpm": 85,
            "valence": 0.5,
            "danceability": 0.4,
            "acousticness": 0.7,
            "popularity": 50,
            "release_decade": 2000,
            "mood_tag": "chill",
            "instrumentalness": 0.6,
            "vocal_presence": 0.4,
            "brightness": 0.4,
        },
    ]


def test_curate_setlist_returns_contract_payload():
    payload = curate_setlist(agent2_payload=_agent2_payload(), songs=_songs(), k=2)

    assert payload["schema_version"] == "1.0"
    assert payload["trace_id"] == "trace-xyz"
    assert isinstance(payload["setlist"], list)
    assert len(payload["setlist"]) == 2
    assert isinstance(payload["explanations"], list)
    assert payload["profile_echo"]["favorite_genre"] == "pop"
    assert payload["retrieval"]["retriever"] in {"token-overlap", "gemini-semantic"}
    assert payload["retrieval"]["candidates_after"] >= 2


def test_setlist_curator_wrapper_works():
    curator = SetlistCurator()
    payload = curator.curate(agent2_payload=_agent2_payload(), songs=_songs(), k=1)

    assert len(payload["setlist"]) == 1
    assert payload["setlist"][0]["rank"] == 1


def test_curate_setlist_honors_avoid_genres_in_retrieval_pool():
    payload_in = _agent2_payload()
    payload_in["profile"]["avoid_genres"] = ["pop"]

    # Use simple (non-agentic) path so avoid_genres isn't cleared by retry logic.
    payload = curate_setlist(agent2_payload=payload_in, songs=_songs(), k=1, agentic=False)

    assert payload["retrieval"]["filtered_avoid_genres"] >= 1
    # With pop avoided, the remaining candidate should come from non-pop entries.
    assert payload["setlist"][0]["title"] == "Track Two"
