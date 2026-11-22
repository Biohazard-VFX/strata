from pathlib import Path


def test_documented_beauty_supervision_matches_final_layer_rule() -> None:
    training = Path(__file__).resolve().parents[2]
    readme = (training / "README.md").read_text()
    assert "Beauty supervision composites the final K layers" in readme
    assert "Raw C beauty stays separate as the reconstruction target" not in readme
