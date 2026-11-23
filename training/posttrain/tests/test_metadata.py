import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from strata_training.data.cache import prepare
from strata_training.data.metadata import CONTRACT, camera_frames, category, intrinsics
from strata_training.io import save_arrays


def test_published_camera_records_use_frame_list_and_native_intrinsics(tmp_path: Path) -> None:
    record = {
        "frame": 1,
        "fx": 100.0,
        "fy": 110.0,
        "cx": 8.0,
        "cy": 4.0,
        "fstop": 0.0,
        "shutter_open": 0.0,
        "shutter_close": 0.0,
    }
    path = tmp_path / "camera.json"
    (tmp_path / "manifest.json").write_text(json.dumps({"contract": CONTRACT}))
    path.write_text(json.dumps({"contract": CONTRACT, "frames": [record]}))
    frame = camera_frames(tmp_path)[1]
    assert_allclose(intrinsics(frame), [[100, 0, 8], [0, 110, 4], [0, 0, 1]])
    assert category(frame) == "sharp"
    record["frame"] = 1.5
    path.write_text(json.dumps({"contract": CONTRACT, "frames": [record]}))
    with pytest.raises(ValueError, match="identifiers"):
        camera_frames(tmp_path)


@pytest.mark.parametrize("protected", [None, "root", "shot"])
def test_native_parts_and_role_masks_are_preserved_in_target_cache(
    tmp_path: Path,
    packed_shot: Path,
    protected: str | None,
) -> None:
    if protected:
        path = (packed_shot if protected == "shot" else packed_shot.parent) / "manifest.json"
        manifest: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        manifest["split"] = "test-only"
        path.write_text(json.dumps(manifest))
    entries = prepare(packed_shot.parent, tmp_path / "targets")
    assert len(entries) == 2
    obsolete: dict[str, Any] = {}
    for entry in entries:
        assert entry["test_only"] == (protected is not None)
        assert entry["split"] == ("test" if protected else "train")
        assert not entry["compact_scene_only"]
        with np.load(entry["cache"]) as data:
            assert str(data["version"]) == "targets-fable-float-v2"
            assert data["alpha"].shape == (4, 1, 4)
            assert_array_equal(data["foreground"], [[0.5, 0.25, 0.5, 0]])
            assert_array_equal(data["hero"], [[0.25, 0.25, 0, 0]])
            assert_array_equal(data["fur"], [[0.25, 0.25, 0, 0]])
            assert_allclose(data["inverse_depth"][0, 0, :3], 0.5)
            assert data["inverse_depth"][0, 0, 3] == 0
            assert len(str(data["content_identity"])) == 64
            obsolete = {k: data[k] for k in data.files if k != "content_identity"}
    save_arrays(Path(entries[0]["cache"]), **obsolete)
    prepare(packed_shot.parent, tmp_path / "targets")
    with np.load(entries[0]["cache"]) as rebuilt:
        assert "content_identity" in rebuilt


@pytest.mark.parametrize("version", ["0.1.0", "0.2.0", "0.3.0", "0.5.0"])
@pytest.mark.parametrize("record", ["manifest", "camera"])
def test_native_preparation_refuses_other_contracts(
    tmp_path: Path,
    packed_shot: Path,
    version: str,
    record: str,
) -> None:
    path = packed_shot / f"{record}.json"
    value = json.loads(path.read_text())
    value["contract"] = f"strata-dataset-{version}"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsupported"):
        prepare(packed_shot.parent, tmp_path / "targets")
    assert not list((tmp_path / "targets").glob("*.npz"))


def test_native_preparation_refuses_undecodable_declared_ids(
    tmp_path: Path,
    packed_shot: Path,
) -> None:
    path = packed_shot / "manifest.json"
    value = json.loads(path.read_text())
    value["instance_ids"][0]["first_id"] = 32768
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsupported dataset records"):
        prepare(packed_shot.parent, tmp_path / "targets")
