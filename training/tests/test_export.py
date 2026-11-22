"""Independent host-only checks for dynamic RoPE and exported precision constants."""

from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import onnx
import torch
from numpy.typing import NDArray
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator
from strata_train._external import resolve
from strata_train.export import dynamic_rope
from strata_train.fold_constants import fold
from torch import nn

type Evaluate = Callable[[list[str], dict[str, NDArray[np.float16]]], list[NDArray[np.float16]]]


class RopeFactory(Protocol):
    def __call__(self, *, dim: int, normalize_coords: str, dtype: torch.dtype) -> nn.Module: ...


class ExportTests(unittest.TestCase):
    def test_rope_matches_unmodified_timm_at_multiple_aspects(self) -> None:
        module = importlib.import_module("timm.layers.pos_embed_sincos")
        factory = cast("RopeFactory", resolve(module, "RotaryEmbeddingDinoV3"))
        fork_rng = cast(
            "Callable[..., AbstractContextManager[None]]", resolve(torch.random, "fork_rng")
        )
        seed = cast("Callable[[int], object]", resolve(torch, "manual_seed"))
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with fork_rng(devices=[]):
                seed(719)
                rope = factory(dim=64, normalize_coords="max", dtype=torch.float32).eval()
                original = cast("Callable[..., torch.Tensor]", resolve(rope, "get_embed"))
                for shape in ((16, 24), (24, 16), (2, 48)):
                    with self.subTest(shape=shape):
                        torch.testing.assert_close(
                            dynamic_rope(rope, shape, dtype=torch.float32),
                            original(shape=shape, dtype=torch.float32),
                            atol=0,
                            rtol=0,
                        )
        finally:
            torch.set_num_threads(threads)

    def test_constant_folding_preserves_onnx_results(self) -> None:
        weight = numpy_helper.from_array(np.array([1.125, -2.5, 4.75], dtype=np.float32), "weight")
        image = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [3])
        output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [3])
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Cast", ["weight"], ["half"], to=onnx.TensorProto.FLOAT16),
                onnx.helper.make_node("Add", ["x", "half"], ["y"]),
            ],
            "constant_cast",
            [image],
            [output],
            [weight],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 23)])
        save = cast("Callable[[onnx.ModelProto,str],None]", resolve(onnx, "save"))
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            path = Path(directory) / "test.onnx"
            save(model, str(path))
            feed = {"x": np.ones(3, dtype=np.float16)}
            before = cast("Evaluate", resolve(ReferenceEvaluator(str(path)), "run"))(["y"], feed)[0]
            fold(path)
            after = cast("Evaluate", resolve(ReferenceEvaluator(str(path)), "run"))(["y"], feed)[0]
        expected = np.array([2.125, -1.5, 5.75], dtype=np.float16)
        np.testing.assert_array_equal(before, expected)
        np.testing.assert_array_equal(after, expected)


if __name__ == "__main__":
    unittest.main()
