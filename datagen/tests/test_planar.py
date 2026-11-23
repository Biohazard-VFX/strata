"""Planar Core decoding and row-streamed QA against independent full-frame NumPy operations."""

import ctypes as c
import tempfile
import unittest
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import OpenEXR

from strata_datagen import exrcore
from strata_datagen.exr import Deep, read_deep
from strata_datagen.qa.deep import FRONT_SLAB, LAYER_GAP_M, OPAQUE, VISIBLE, flatten
from strata_datagen.render import pack
from tests.fixtures import SKY, Sample, write_shot

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _write(path: Path) -> None:
    channels: dict[str, NDArray[np.generic]] = {}
    for name, dtype in (("A", np.float16), ("R", np.float32), ("Z", np.uint32)):
        cells = np.empty((3, 4), object)
        for pixel in range(cells.size):
            count = 0 if pixel // 4 == 1 else pixel % 4
            cells.flat[pixel] = np.arange(count, dtype=dtype) + pixel if count else None
        channels[name] = cells
    OpenEXR.File(
        {
            "type": OpenEXR.deepscanline,
            "compression": OpenEXR.ZIPS_COMPRESSION,
            "dataWindow": (np.array([-5, -7], np.int32), np.array([-2, -5], np.int32)),
        },
        channels,
    ).write(str(path))


class PlanarTests(unittest.TestCase):
    def test_batch_releases_frame_before_reading_next_deep(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shot = Path(temporary) / "shot"
            write_shot(shot, [[Sample(0.1, 0.5, 2), SKY]] * 2, 2)
            references: list[weakref.ReferenceType[pack.Samples]] = []
            original_samples, original_read = pack.samples, pack.read_deep

            def tracked(deep: Deep) -> pack.Samples:
                value = original_samples(deep)
                references.append(weakref.ref(value))
                return value

            def checked(path: Path) -> Deep:
                self.assertTrue(
                    all(reference() is None for reference in references),
                    "previous frame retained into the next EXR read",
                )
                return original_read(path)

            with (
                patch.object(pack, "samples", side_effect=tracked),
                patch.object(pack, "read_deep", side_effect=checked),
            ):
                pack.pack(shot, range(1, 3))
            self.assertEqual(len(references), 2)
            self.assertTrue(all(reference() is None for reference in references))

    def test_deep_values_and_empty_pixels_match_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deep.exr"
            _write(path)
            reference = OpenEXR.File(str(path), separate_channels=True).parts[0]
            deep = read_deep(path)
            self.assertEqual((deep.height, deep.width), (3, 4))
            np.testing.assert_array_equal(deep.counts, [0, 1, 2, 3, 0, 0, 0, 0, 0, 1, 2, 3])
            for name, channel in reference.channels.items():
                cells = cast("list[NDArray[np.generic] | None]", channel.pixels.ravel().tolist())
                expected = np.concatenate([cell for cell in cells if cell is not None], dtype=np.float32)
                self.assertEqual(deep.channels[name].dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(deep.channels[name], expected.astype(np.float32))

    def test_changed_count_table_is_refused_before_unpacking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deep.exr"
            _write(path)
            original = exrcore.LIBRARY.exr_decoding_run

            def changed(context: c.c_void_p, part: int, pointer: c.c_void_p) -> int:
                code = int(original(context, part, pointer))
                pipeline = c.cast(pointer, c.POINTER(exrcore.Pipeline)).contents
                if pipeline.flags == 5 and pipeline.chunk.y == -7:
                    pipeline.counts[1] += 1
                    pipeline.counts[2] -= 1
                return code

            with (
                patch.object(exrcore.LIBRARY, "exr_decoding_run", side_effect=changed),
                self.assertRaises(ValueError),
            ):
                read_deep(path)

    def test_streamed_flatten_keeps_global_rounding_and_stable_depth_order(self) -> None:
        rng = np.random.default_rng(1927)
        shape = (7, 9)
        counts = rng.integers(0, 512, np.prod(shape), dtype=np.int64)
        counts[shape[1] : 2 * shape[1]] = 0
        counts[[0, -1]] = 0
        size = int(counts.sum())
        depth = rng.integers(1, 7, size).astype(np.float32)
        channels = {name: rng.random(size).astype(np.float32) for name in "RGBA"}
        channels["Z"] = depth
        deep = Deep(shape[1], shape[0], counts, channels)
        pixel = np.repeat(np.arange(counts.size), counts)
        order = np.lexsort((depth, pixel))
        alpha = np.clip(channels["A"][order].astype(np.float64), 0.0, 0.999999)
        logs = np.log1p(-alpha)
        exclusive = np.cumsum(logs) - logs
        starts = np.concatenate([[0], np.cumsum(counts)])[:-1]
        transmitted = np.exp(exclusive - np.repeat(exclusive[starts[counts > 0]], counts[counts > 0]))
        expected = np.zeros((counts.size, 4))
        for i, name in enumerate("RGBA"):
            np.add.at(expected[:, i], pixel, channels[name][order] * transmitted)
        z = depth[order].astype(np.float64)
        near = np.full(counts.size, np.inf)
        np.minimum.at(near, pixel, z)
        front = z <= near[pixel] * (1.0 + FRONT_SLAB) + LAYER_GAP_M
        clear = np.zeros(counts.size)
        np.add.at(clear, pixel[front], logs[front])
        coverage = 1.0 - np.exp(clear)
        partial = (coverage > VISIBLE) & (coverage < OPAQUE) & (expected[:, 3] > VISIBLE)
        result = flatten(deep)
        np.testing.assert_array_equal(result.rgba, expected.reshape(*shape, 4))
        np.testing.assert_array_equal(result.alpha, expected[:, 3].reshape(shape))
        np.testing.assert_array_equal(result.partial, partial.reshape(shape))
