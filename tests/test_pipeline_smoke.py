from src.recommender import load_songs
from src.orchestrator import run_pipeline


def test_full_pipeline_smoke_local_backend():
    songs = load_songs("data/songs.csv")
    result = run_pipeline(
        user_message="chill study lofi please",
        songs=songs,
        k=3,
        agent1_backend="local",
    )

    assert result["agent1"]["schema_version"] == "1.0"
    assert result["agent2"]["schema_version"] == "1.0"
    assert result["agent3"]["schema_version"] == "1.0"
    assert result["agent4"]["schema_version"] == "1.0"
    assert len(result["agent3"]["setlist"]) == 3
