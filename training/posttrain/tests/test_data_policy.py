import hashlib
from pathlib import Path

import pytest

from strata_training.eval.protocol import entries_for
from strata_training.io import write_json


@pytest.mark.parametrize("root_marker", [False, True])
@pytest.mark.parametrize("legacy_marker", [False, True])
def test_test_only_manifests_cannot_be_relabelled_for_training_or_validation(
    tmp_path: Path,
    root_marker: bool,
    legacy_marker: bool,
) -> None:
    dataset = tmp_path / "dataset"
    shot = dataset / "shot"
    manifest = shot / "manifest.json"
    marker = {"eval_only": True} if legacy_marker else {"split": "test-only"}
    write_json(manifest, {"split": "train"} if root_marker else marker)
    if root_marker:
        write_json(dataset / "manifest.json", marker)
    entry = {"id": "frame", "shot": "shot", "split": "train", "manifest": str(manifest)}
    index = tmp_path / "index.json"
    write_json(index, [entry])
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    training, testing = tmp_path / "train.json", tmp_path / "test.json"
    write_json(training, {"index_sha256": digest, "train": ["frame"], "test": []})
    write_json(testing, {"index_sha256": digest, "train": [], "test": ["frame"]})
    with pytest.raises(ValueError, match="test-only"):
        entries_for(index, training, "train")
    with pytest.raises(ValueError, match="test-only"):
        entries_for(index, testing, "test", purpose="validation")
    assert entries_for(index, testing, "test") == [entry]
