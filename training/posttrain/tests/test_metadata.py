import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from strata_training.data.exr import exr, read_instances, read_rgb
from strata_training.data.metadata import CONTRACT, camera_frames, category, intrinsics


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


def test_beauty_C_and_true_groom_id_are_read_independently_of_part_order(tmp_path: Path) -> None:
    ids = np.array([[3, 1], [2, 3]], dtype=np.float32)
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
    path = tmp_path / "beauty.exr"
    with exr.File(parts) as file:
        file.write(str(path))
    assert_allclose(read_rgb(path)[:, 0, 0], [1, 0.5, 0.25])
    assert_array_equal(read_instances(path), [[3, 1], [2, 3]])
