import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from strata_training.data.cache import display_rgb
from strata_training.data.encoder_input import display, normalize
from strata_training.data.feature_cache import read_base
from strata_training.io import save_arrays

VALUES = [0, 0.001, 0.18, 0.6 / 0.72, 1, 1.4, 10, 64, 128, 1000, -1, np.nan, np.inf, -np.inf]
EXPECTED = [
    0,
    0.00930240005,
    0.395302713,
    0.797737718,
    0.850128949,
    0.907322526,
    0.989884555,
    0.998465478,
    0.999234796,
    0.999902308,
    0,
    0,
    1,
    0,
]


def test_fable_fixed_points_and_float_cache() -> None:
    image = np.broadcast_to(np.array(VALUES, np.float32), (3, 1, len(VALUES))).copy()
    encoded = display_rgb(image, "Linear Rec.709")
    assert encoded.dtype == np.float32
    np.testing.assert_allclose(encoded[0, 0], EXPECTED, rtol=0, atol=6e-8)
    assert np.any(np.abs(encoded * 255 - np.round(encoded * 255)) > 0.1)
    with pytest.raises(ValueError, match="float32"):
        normalize(torch.ones((3, 1, 1), dtype=torch.uint8))


def test_exposure_is_fixed_across_frames_and_crops() -> None:
    image = torch.tensor(VALUES, dtype=torch.float32).expand(2, 3, 1, -1).clone()
    image[1, :, :, 0] = 1000
    for stops in (-4, 0, 2):
        mapped = display(image, "Linear Rec.709", stops)
        torch.testing.assert_close(mapped[0, :, :, 1:], mapped[1, :, :, 1:], rtol=0, atol=0)
        torch.testing.assert_close(
            mapped[..., 1:], display(image[..., 1:], "Linear Rec.709", stops), rtol=0, atol=0
        )
        torch.testing.assert_close(
            mapped, display(image * 2.0**stops, "Linear Rec.709"), rtol=0, atol=0
        )


def test_pre_shoulder_feature_cache_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "base.npz"
    save_arrays(
        path,
        version=np.array("features-metric-v2"),
        inverse_depth=np.ones((1, 1)),
        metric_scale=np.array(1),
    )
    with pytest.raises(ValueError, match="version mismatch"):
        read_base(path)


@pytest.mark.parametrize("version", [None, "targets-content-v1"])
@pytest.mark.parametrize("consumer", ["extract", "baselines", "frame"])
def test_stale_target_caches_are_rejected(
    tmp_path: Path, version: str | None, consumer: str
) -> None:
    from strata_training.baselines.prepare import prepare_baselines
    from strata_training.data.dataset import Frame
    from strata_training.io import write_json
    from strata_training.model.features import extract

    cache = tmp_path / "target.npz"
    values = {"srgb": np.ones((3, 1, 1), np.float32)}
    if version is not None:
        values["version"] = np.array(version)
    save_arrays(cache, **values)
    index = tmp_path / "index.json"
    entry = {"id": "frame", "cache": str(cache)}
    write_json(index, [entry])

    def consume() -> None:
        if consumer == "extract":
            extract(index, tmp_path / "features", "test", device="cpu")
        elif consumer == "baselines":
            prepare_baselines(index, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
        else:
            Frame(entry, tmp_path)

    with pytest.raises(ValueError, match="Target cache version mismatch"):
        consume()


def test_encoder_stays_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import import_module
    from types import SimpleNamespace
    from typing import Any

    from torch import nn

    from strata_training.model.dp2 import DeepDepthPro

    upstream: Any = import_module("depth_pro_2.model")
    base = nn.Module()
    base.encoder = nn.Linear(3, 3)
    base.decoder = nn.Linear(3, 3)
    base.head = nn.Sequential(nn.Identity(), nn.Identity(), nn.Conv2d(2, 1, 1))

    def construct(**_kwargs: Any) -> nn.Module:
        return base

    def checkpoint(_uri: str) -> dict[str, torch.Tensor]:
        return base.state_dict()

    config = SimpleNamespace(
        backbone_preset="test",
        reassembled_blocks=[],
        cross_frame_blocks=[],
        dims_reassemble=[],
        dim_decoder=4,
        checkpoint_uri="test",
    )
    monkeypatch.setattr(upstream, "PRESETS", {"test": config})
    monkeypatch.setattr(upstream, "DepthPro2", construct)
    monkeypatch.setattr(upstream, "load_checkpoint", checkpoint)
    model = DeepDepthPro("test").train()
    assert not model.base.encoder.training
    assert all(not p.requires_grad for p in model.base.encoder.parameters())
    assert all(p.requires_grad for p in model.base.decoder.parameters())
    assert all(p.requires_grad for p in model.deep_head.output.parameters())


def parity(fixture: Path, oracle: Path) -> None:
    knee = np.float32(0.6 / 0.72)
    values = np.resize(
        np.concatenate(
            (
                np.array([*VALUES, -1000, -0.001, np.inf, -np.inf, np.nan], np.float32),
                np.linspace(-1, 4, 480, dtype=np.float32),
                np.geomspace(1e-9, 1e30, 480, dtype=np.float32),
                np.array(
                    [np.nextafter(knee, np.float32(0)), knee, np.nextafter(knee, np.float32(2))]
                ),
            )
        ),
        1023,
    )
    image = torch.as_tensor(values.copy()).expand(3, 1, -1)
    maximum = 0.0
    exposures = (-4, 0, 2, 0.1, -0.3)
    encoded = np.stack([display(image, "Linear Rec.709", stops).numpy() for stops in exposures])
    save_arrays(oracle, linear=image.numpy(), exposures=np.array(exposures), encoded=encoded)
    for stops in exposures:
        result = subprocess.run(
            [str(fixture), str(stops)],
            input=" ".join(map(str, values)).encode(),
            capture_output=True,
            check=True,
        )
        actual = np.frombuffer(result.stdout, np.float32).reshape(3, 32, 1024)
        expected = torch.nn.functional.pad(
            normalize(display(image, "Linear Rec.709", stops)), (0, 1, 0, 31), mode="replicate"
        ).numpy()
        error = float(np.max(np.abs(actual - expected)))
        maximum = max(maximum, error)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    print(f"C/training encoder parity: 491520 values, max_abs={maximum:.9g}, tolerance=0")


if __name__ == "__main__":
    parity(Path(sys.argv[1]), Path(sys.argv[2]))
