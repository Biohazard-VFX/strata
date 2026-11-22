from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from strata_training.data.exr import (
    decode_instance_ids,
    exr,
    read_deep,
    read_roles,
    read_samples,
)
from strata_training.data.targets import reduce_samples
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
    assert_array_equal(decode_instance_ids(arrays["instance_id"], arrays["A"]), [1, 3, 2, 0])
    arrays["Z"][0] = 99
    assert expected["Z"][0] == 8


def test_default_raw_reader_and_existing_deep_and_roles_are_compatible(tmp_path: Path) -> None:
    path = tmp_path / "deep.exr"
    write_samples(path)
    arrays, counts, shape = read_samples(path)
    assert set(arrays) == {"Z", "A", "R", "G", "B", "instance_id"}
    expected = reduce_samples(
        arrays["Z"], arrays["A"], np.stack([arrays[c] for c in "RGB"], axis=1), counts, shape
    )
    actual = read_deep(path)
    for name, values in expected.arrays().items():
        assert_array_equal(actual.arrays()[name], values)
    roles = read_roles(path)
    assert_array_equal(roles["fur"], [[True, False], [False, False]])
    assert_array_equal(roles["foreground"], [[True, False], [False, False]])


def test_legacy_missing_ids_are_not_fabricated(tmp_path: Path) -> None:
    path = tmp_path / "legacy.exr"
    write_samples(path, ids=False)
    with pytest.raises(KeyError, match="instance_id"):
        read_samples(path)
    with pytest.raises(KeyError, match="instance_id"):
        read_roles(path)
    arrays, counts, _ = read_samples(path, "ZARGB")
    assert "instance_id" not in arrays
    assert_array_equal(counts, [3, 0, 1, 0])
    assert read_deep(path).alpha[0, 0, 0] == 0.25


@pytest.mark.parametrize("invalid", [-1, np.inf, np.nan])
def test_invalid_premultiplied_ids_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        decode_instance_ids(np.array([invalid]), np.ones(1))


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
