import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.retrieval import _token_overlap_confidence, retrieve_candidates


def _songs() -> List[Dict[str, Any]]:
    return [
        {
            "id": 1, "title": "Gym Hero", "artist": "Max Pulse",
            "genre": "pop", "mood": "intense", "mood_tag": "aggressive",
        },
        {
            "id": 2, "title": "Rainy Window Notes", "artist": "Soft Keys",
            "genre": "lofi", "mood": "chill", "mood_tag": "dreamy",
        },
        {
            "id": 3, "title": "Midnight Fade", "artist": "Dark Echo",
            "genre": "indie", "mood": "moody", "mood_tag": "dreamy",
        },
        {
            "id": 4, "title": "Storm Runner", "artist": "Voltline",
            "genre": "metal", "mood": "intense", "mood_tag": "aggressive",
        },
    ]


def _agent2_payload(genre="pop", mood="intense", avoid=None) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "trace_id": "test-trace",
        "profile": {
            "favorite_genre": genre,
            "favorite_mood": mood,
            "target_energy": 0.8,
            "likes_acoustic": False,
            "avoid_genres": avoid or [],
        },
        "constraints": {},
        "request_summary": f"{mood} {genre}",
    }


# ---------------------------------------------------------------------------
# Token-overlap path (no api_key)
# ---------------------------------------------------------------------------

def test_token_overlap_used_when_no_api_key():
    _, debug = retrieve_candidates(_agent2_payload(), _songs(), top_n=3)
    assert debug["retriever"] == "token-overlap"


def test_token_overlap_returns_correct_count():
    results, debug = retrieve_candidates(_agent2_payload(), _songs(), top_n=2)
    assert len(results) == 2
    assert debug["candidates_after"] == 2


def test_token_overlap_confidence_is_in_debug():
    _, debug = retrieve_candidates(_agent2_payload(), _songs(), top_n=3)
    assert "retrieval_confidence" in debug
    assert 0.0 <= debug["retrieval_confidence"] <= 1.0


def test_token_overlap_confidence_higher_for_strong_match():
    # "intense pop" query matches Gym Hero directly on genre + mood tokens
    _, debug_strong = retrieve_candidates(
        _agent2_payload(genre="pop", mood="intense"), _songs(), top_n=4
    )
    # "balanced jazz" query has no matching songs — weaker signal
    _, debug_weak = retrieve_candidates(
        _agent2_payload(genre="jazz", mood="balanced"), _songs(), top_n=4
    )
    assert debug_strong["retrieval_confidence"] >= debug_weak["retrieval_confidence"]


def test_avoid_genres_filtered_from_token_overlap():
    payload = _agent2_payload(avoid=["metal"])
    results, debug = retrieve_candidates(payload, _songs(), top_n=10)

    genres = [s["genre"] for s in results]
    assert "metal" not in genres
    assert debug["filtered_avoid_genres"] == 1


def test_token_overlap_returns_only_non_avoided_genres():
    # Avoid pop, lofi, indie — only metal remains
    payload = _agent2_payload(avoid=["pop", "lofi", "indie"])
    results, debug = retrieve_candidates(payload, _songs(), top_n=5)

    assert all(s["genre"] == "metal" for s in results)
    assert debug["retrieval_fallback"] is False  # metal still scores on "intense" token


def test_token_overlap_fallback_when_no_songs_remain():
    # Avoid all genres → scored is empty, fallback path triggers
    payload = _agent2_payload(avoid=["pop", "lofi", "indie", "metal"])
    results, debug = retrieve_candidates(payload, _songs(), top_n=5)

    assert results == []
    assert debug["retrieval_fallback"] is True
    assert debug["retrieval_confidence"] == 0.0


# ---------------------------------------------------------------------------
# Gemini path (mocked)
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str):
    """Build a mock LLM that returns a fixed string."""
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


def test_gemini_path_used_when_api_key_provided():
    gemini_response = json.dumps({"ids": [1, 3], "confidence": 0.88})

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_mock_llm_response(gemini_response)), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        results, debug = retrieve_candidates(
            _agent2_payload(), _songs(), top_n=2,
            user_message="hype workout songs", api_key="fake-key",
        )

    assert debug["retriever"] == "gemini-semantic"
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 3


def test_gemini_confidence_parsed_from_response():
    gemini_response = json.dumps({"ids": [2, 3], "confidence": 0.76})

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_mock_llm_response(gemini_response)), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        _, debug = retrieve_candidates(
            _agent2_payload(), _songs(), top_n=3,
            user_message="late night chill", api_key="fake-key",
        )

    assert debug["retrieval_confidence"] == 0.76


def test_gemini_falls_back_to_token_overlap_on_invalid_json():
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_mock_llm_response("not valid json at all")), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        _, debug = retrieve_candidates(
            _agent2_payload(), _songs(), top_n=2,
            user_message="something", api_key="fake-key",
        )

    assert debug["retriever"] == "token-overlap"


def test_gemini_falls_back_to_token_overlap_on_exception():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("API unreachable")

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_llm), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        _, debug = retrieve_candidates(
            _agent2_payload(), _songs(), top_n=2,
            user_message="something", api_key="fake-key",
        )

    assert debug["retriever"] == "token-overlap"


def test_gemini_avoid_genres_not_in_results():
    # Gemini returns id=4 (metal) but metal is in avoid_genres
    gemini_response = json.dumps({"ids": [4, 1], "confidence": 0.7})

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_mock_llm_response(gemini_response)), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        results, _ = retrieve_candidates(
            _agent2_payload(avoid=["metal"]), _songs(), top_n=3,
            user_message="intense songs", api_key="fake-key",
        )

    # id=4 is metal and was passed to avoid set before Gemini call, so catalog
    # sent to Gemini excluded it. Even if Gemini somehow returned it, song_by_id
    # lookup will still resolve — but the catalog filtering means Gemini won't see it.
    genres = [s["genre"] for s in results]
    assert "metal" not in genres


def test_gemini_bare_array_response_still_works():
    """Gemini ignores instructions and returns a bare array — should still parse."""
    gemini_response = "[2, 3]"

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=_mock_llm_response(gemini_response)), \
         patch("langchain_core.messages.HumanMessage", side_effect=lambda content: content):
        results, debug = retrieve_candidates(
            _agent2_payload(), _songs(), top_n=3,
            user_message="chill vibes", api_key="fake-key",
        )

    assert debug["retriever"] == "gemini-semantic"
    assert any(s["id"] in {2, 3} for s in results)


# ---------------------------------------------------------------------------
# _token_overlap_confidence unit tests
# ---------------------------------------------------------------------------

def test_confidence_is_zero_for_empty_scored():
    assert _token_overlap_confidence([], ["pop", "intense"]) == 0.0


def test_confidence_is_zero_for_empty_tokens():
    assert _token_overlap_confidence([(3.0, {})], []) == 0.0


def test_confidence_clamps_to_one():
    # Artificially high score should clamp at 1.0
    assert _token_overlap_confidence([(9999.0, {})], ["a"]) == 1.0


def test_confidence_is_between_zero_and_one_for_normal_input():
    scored = [(4.5, {}), (2.0, {}), (0.5, {})]
    tokens = ["pop", "intense", "workout"]
    conf = _token_overlap_confidence(scored, tokens)
    assert 0.0 <= conf <= 1.0
