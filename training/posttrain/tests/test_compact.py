import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from strata_dataset import sampling

from strata_training.data.cache import prepare
from strata_training.data.exr import exr, read_rgb
from strata_training.data.metadata import CONTRACT
from strata_training.io import write_json
from strata_training.train import train
from strata_training.types import Array


def test_compact_frame_without_raw_deep_preserves_targets_and_denoised_part(tmp_path: Path) -> None:
    shot = tmp_path / "dataset" / "shot"
    (shot / "layers").mkdir(parents=True)
    (shot / "beauty").mkdir()
    (shot / "manifest.json").write_text(
        json.dumps(
            {
                "contract": CONTRACT,
                "shot_id": "shot",
                "background": {"instance_id": 0},
                "seed": 0,
                "render": {"frames": [2, 2], "sampling_seed": sampling.schedule(0, (2,))},
            }
        )
    )
    (shot / "camera.json").write_text(
        json.dumps(
            {
                "contract": CONTRACT,
                "frames": [
                    {
                        "frame": 2,
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
                ],
            }
        )
    )
    channels: dict[str, Array] = {}
    for k in range(4):
        channels[f"L{k}.invdepth"] = np.full((2, 2), 0.5 if k == 0 else 0, np.float32)
        for c in "ARGB":
            channels[f"L{k}.{c}"] = np.full((2, 2), 0.25 if k == 0 else 0, np.float16)
    channels |= {
        f"roles.{role}": np.full((2, 2), 0.25, np.float32) for role in ("foreground", "hero", "fur")
    }
    exr.File({}, channels).write(str(shot / "layers/0002.exr"))
    parts = [
        exr.Part({"name": "C"}, {c: np.full((2, 2), 0.7, np.float32) for c in "RGB"}),
        exr.Part(
            {"name": "denoised"}, {f"denoised.{c}": np.full((2, 2), 0.5, np.float32) for c in "RGB"}
        ),
    ]
    exr.File(parts).write(str(shot / "beauty/0002.exr"))
    entries = prepare(tmp_path / "dataset", tmp_path / "targets")
    assert len(entries) == 1
    assert entries[0]["frame"] == 2
    with np.load(entries[0]["cache"]) as target:
        assert_allclose(target["inverse_depth"][0], 0.5)
        assert_allclose(target["alpha"][0], 0.25)
        assert_allclose(target["rgb"][0], 0.25)
        assert_allclose(target["denoised"], 0.5)
        assert_allclose(target["beauty"], 0.7, atol=0.0003)
        assert_allclose(target["fur"], 0.25)
    assert_allclose(read_rgb(shot / "beauty/0002.exr", part_name="denoised"), 0.5)
    assert prepare(tmp_path / "dataset", tmp_path / "targets") == entries
    index = tmp_path / "targets/index.json"
    protocol = tmp_path / "protocol.json"
    write_json(
        protocol,
        {
            "train": [entries[0]["id"]],
            "version": 2,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    )
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "index": str(index),
            "protocol": str(protocol),
            "features": str(tmp_path / "features"),
            "preset": "depthpro2-h",
            "layers": 4,
            "mode": "head",
            "input_mode": "cached",
            "seed": 1,
            "crop": 32,
            "halo": 0,
            "batch_size": 1,
            "steps": 1,
            "learning_rate": 4e-4,
            "output": str(tmp_path / "run"),
            "device": "cpu",
        },
    )
    entries[0]["compact_scene_only"] = True
    write_json(index, entries)
    protocol_data = json.loads(protocol.read_text())
    protocol_data["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    write_json(protocol, protocol_data)
    with pytest.raises(ValueError, match="omit sky colour/coverage"):
        train(config)
