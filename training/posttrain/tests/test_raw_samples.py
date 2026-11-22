from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from strata_training.data.exr import (
    exr,
    read_samples,
)
from strata_training.io import atomic_path
from strata_training.types import Array


def write_samples(path: Path, *, ids: bool = True) -> dict[str, Array]:
    values = {
        "Z": np.array([8, 2, 4, 3], dtype=np.float32),
        "ZBack": np.array([9, 2.5, 4.5, 3], dtype=np.float32),
        "A": np.array([1, 0.25, 0.5, 0], dtype=np.float32),
        "R": np.array([4, 0.5, 1, 0], dtype=np.float32),
        "G": np.array([2, 0.25, 0.5, 0], dtype=np.float32),
        "B": np.array([1, 0, 0.25, 0], dtype=np.float32),
    }
    values["coverage"] = values["A"].copy()
    if ids:
        values["instance_id"] = np.array([1, 0.75, 1, 0], dtype=np.float32)
    channels: dict[str, Array] = {}
    for name, samples in values.items():
        pixels = np.empty((2, 2), dtype=object)
        pixels[0, 0], pixels[0, 1] = samples[:3], None
        pixels[1, 0], pixels[1, 1] = samples[3:], np.empty(0, dtype=np.float32)
        channels[name] = pixels
    with (
        atomic_path(path) as temporary,
        exr.File({"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels) as file,
    ):
        file.write(str(temporary))
    return values


def test_raw_samples_own_unsorted_channels_counts_and_optional_back_depth(tmp_path: Path) -> None:
    path = tmp_path / "deep.exr"
    expected = write_samples(path)
    arrays, counts, shape = read_samples(path, tuple(expected))
    path.unlink()
    assert shape == (2, 2)
    assert counts.flags.owndata
    assert_array_equal(counts, [3, 0, 1, 0])
    for name, values in expected.items():
        assert arrays[name].flags.owndata
        assert arrays[name].dtype == np.float64
        assert_array_equal(arrays[name], values)
    arrays["Z"][0] = 99
    assert expected["Z"][0] == 8


def test_native_raw_and_sidecar_samples_preserve_unlabelled_roles(
    tmp_path: Path,
    packed_shot: Path,
    math_device: str,
) -> None:
    import torch
    from strata_dataset.samples import read

    from strata_training.augment.deep import load_samples
    from strata_training.augment.deep import read_samples as read_native
    from strata_training.augment.torch_deep import TensorSamples
    from strata_training.data.compact import read_layers
    from strata_training.data.online import prepare_sample

    source = read(packed_shot / "samples/shot_0001").arrays
    deep = read_native(packed_shot / "deep/0001.exr")
    assert deep.ids[0] == -1
    assert_array_equal(deep.roles[0], [0.5, 0.25, 0.25])
    assert deep.ids[2] == 42
    assert_array_equal(deep.roles[2], [0.25, 0.25, 0.25])
    assert deep.ids[4] == 3
    assert_array_equal(deep.roles[4], [0.5, 0, 0])
    entry = {"id": "shot_0001", "frame": 1, "manifest": str(packed_shot / "manifest.json")}
    prepare_sample(entry, None, tmp_path / "samples")
    copied = load_samples(tmp_path / "samples/deep")
    for name in ("z", "alpha", "rgb", "ids", "roles", "offsets"):
        assert_array_equal(getattr(copied, name), source[name])
        assert_array_equal(getattr(deep, name), source[name])
    expected = read_layers(packed_shot / "layers/0001.exr")
    tensors = TensorSamples(
        deep.shape,
        *(
            torch.tensor(getattr(deep, name), device=math_device)
            for name in ("counts", "z", "alpha", "rgb", "ids", "roles")
        ),
    )
    from dataclasses import replace

    unlabelled = replace(deep, ids=np.full_like(deep.ids, -1))
    unlabelled_tensors = replace(tensors, ids=torch.full_like(tensors.ids, -1))
    for actual in (
        deep.targets().arrays(),
        tensors.targets(),
        unlabelled.targets().arrays(),
        unlabelled_tensors.targets(),
    ):
        for name, value in expected.arrays().items():
            a = actual[name]
            np.testing.assert_allclose(
                a.cpu().numpy() if isinstance(a, torch.Tensor) else a, value, atol=2e-6, rtol=2e-6
            )
    assert deep.targets().inverse_depth[0, 0, 0] == 0.5


def test_raw_reader_handles_a_completely_empty_image(tmp_path: Path) -> None:
    path = tmp_path / "empty.exr"
    names = ("Z", "A", "R", "G", "B", "instance_id")
    channels = {name: np.empty((1, 2), dtype=object) for name in names}
    for pixels in channels.values():
        for x in range(2):
            pixels[0, x] = np.empty(0, dtype=np.float32)
    with (
        atomic_path(path) as temporary,
        exr.File({"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels) as file,
    ):
        file.write(str(temporary))
    arrays, counts, shape = read_samples(path)
    path.unlink()
    assert shape == (1, 2)
    assert_array_equal(counts, [0, 0])
    for values in arrays.values():
        assert values.flags.owndata
        assert values.shape == (0,)
        assert values.dtype == np.float64
