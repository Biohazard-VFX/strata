from pathlib import Path


def test_documented_beauty_supervision_matches_final_layer_rule() -> None:
    training = Path(__file__).resolve().parents[2]
    readme = (training / "README.md").read_text()
    assert "Beauty supervision composites the final K layers" in readme
    assert "Raw C beauty stays separate as the reconstruction target" not in readme


def test_documented_preparation_platform_requirements() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    parallel = readme.split("### Parallel CPU preparation", 1)[1].split("###", 1)[0]
    assert "TMPDIR" in parallel
    assert "Unix sockets" in parallel
    assert "exFAT/FAT" in parallel
    assert "60 seconds" in parallel
    cache = " ".join(readme.split("## Target cache maintenance", 1)[1].split())
    assert "a memo record with a missing or different boot ID is rehashed" in cache
    assert "preparation requires Linux `/proc`" in cache
