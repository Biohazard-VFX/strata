import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from strata_training.data.cache import prepare
from strata_training.data.exr import exr, read_roles, write_deep
from strata_training.data.metadata import CONTRACT, camera_frames, category, intrinsics
from strata_training.types import Layers


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
    path.write_text(json.dumps({"contract": CONTRACT, "frames": [record]}))
    frame = camera_frames(tmp_path)[1]
    assert_allclose(intrinsics(frame), [[100, 0, 8], [0, 110, 4], [0, 0, 1]])
    assert category(frame) == "sharp"
    record["frame"] = 1.5
    path.write_text(json.dumps({"contract": CONTRACT, "frames": [record]}))
    with pytest.raises(ValueError, match="identifiers"):
        camera_frames(tmp_path)


def test_native_parts_and_role_masks_are_preserved_in_target_cache(tmp_path: Path) -> None:
    shot = tmp_path / "dataset" / "shot"
    (shot / "beauty").mkdir(parents=True)
    (shot / "deep").mkdir()
    (shot / "manifest.json").write_text(json.dumps({"contract": CONTRACT, "shot_id": "shot"}))
    record = {
        "frame": 1,
        "width": 2,
        "height": 2,
        "fx": 10,
        "fy": 10,
        "cx": 1,
        "cy": 1,
        "fstop": 0,
        "shutter_open": 0,
        "shutter_close": 0,
    }
    (shot / "camera.json").write_text(json.dumps({"contract": CONTRACT, "frames": [record]}))
    ids = np.array([[3, 1000], [2, 1]], dtype=np.float32)
    parts = [
        exr.Part({"name": "instance_id"}, {"instance_id.instance_id": ids}),
        exr.Part(
            {"name": "C"},
            {
                c: np.full((2, 2), v, dtype=np.float32)
                for c, v in zip("RGBA", (1, 0.5, 0.25, 1), strict=True)
            },
        ),
    ]
    path = shot / "beauty/0001.exr"
    with exr.File(parts) as file:
        file.write(str(path))
    alpha = np.ones((1, 2, 2), dtype=np.float32)
    rgb = np.zeros((1, 3, 2, 2), dtype=np.float32)
    rgb[0] = np.array([1, 0.5, 0.25], dtype=np.float32)[:, None, None]
    deep_path = shot / "deep/0001.exr"
    write_deep(deep_path, Layers(0.5 * alpha, alpha, rgb))
    with exr.File(str(deep_path), separate_channels=True) as file:
        channels = {k: v.pixels.copy() for k, v in file.parts[0].channels.items()}
    samples = np.empty((2, 2), dtype=object)
    for y, x in np.ndindex(2, 2):
        samples[y, x] = np.array([ids[y, x]], dtype=np.float32)
    channels["instance_id"] = samples
    with exr.File(
        {"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels
    ) as file:
        file.write(str(deep_path))
    entries = prepare(tmp_path / "dataset", tmp_path / "targets")
    with np.load(entries[0]["cache"]) as data:
        assert data["alpha"].shape == (4, 2, 2)
        assert_allclose(data["beauty"][:, 0, 0], [1, 0.5, 0.25])
        assert_allclose(data["inverse_depth"][0], 0.5)
        assert_array_equal(data["fur"], [[True, False], [False, False]])
        assert_array_equal(data["foreground"], [[True, False], [True, False]])


def test_deep_roles_decode_premultiplied_ids_and_ignore_hidden_samples(tmp_path: Path) -> None:
    channels = {k: np.empty((1, 4), dtype=object) for k in ("Z", "A", "instance_id")}
    rows = [
        ([4, 2], [1, 0.25], [1000, 0.75]),
        ([2, 4], [1, 1], [1, 3]),
        ([4, 2], [1, 0.5], [1, 1]),
        ([2], [0], [0]),
    ]
    for x, row in enumerate(rows):
        for key, values in zip(channels, row, strict=True):
            channels[key][0, x] = np.array(values, dtype=np.float32)
    path = tmp_path / "deep.exr"
    with exr.File(
        {"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels
    ) as file:
        file.write(str(path))
    roles = read_roles(path)
    assert_array_equal(roles["fur"], [[True, False, False, False]])
    assert_array_equal(roles["foreground"], [[True, False, True, False]])
