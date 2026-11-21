"""Exact-rewrite checks for the export optimizer, runnable without a GPU."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import cast

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from strata_train._external import resolve
from strata_train.optimize import fuse_rotary, rewrite

type Run = Callable[[list[str] | None, dict[str, NDArray[np.float32]]], list[NDArray[np.float32]]]


def evaluate(model: onnx.ModelProto, x: NDArray[np.float32]) -> NDArray[np.float32]:
    run = cast("Run", resolve(ReferenceEvaluator(model), "run"))
    return run(None, {"x": x})[0]


def model(nodes: list[onnx.NodeProto], weights: list[onnx.TensorProto]) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "optimize",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4, 3, 5])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
        weights,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


class RotaryEmbedding(OpRun):
    op_domain = "com.microsoft"

    def _run(
        self,
        x: NDArray[np.float32],
        position: NDArray[np.int64],
        cos: NDArray[np.float32],
        sin: NDArray[np.float32],
        interleaved: int = 0,
        num_heads: int | None = None,
        rotary_embedding_dim: int | None = None,
    ) -> tuple[NDArray[np.float32]]:
        assert not interleaved and not num_heads and not rotary_embedding_dim
        half = x.shape[-1] // 2
        rows = slice(int(position[0]), int(position[0]) + x.shape[2])
        c, s = cos[rows], sin[rows]
        first, second = x[..., :half], x[..., half:]
        rotated = np.concatenate([first * c - second * s, second * c + first * s], -1)
        return (rotated.astype(np.float32),)


def rope_model() -> onnx.ModelProto:
    f16, f32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    nodes = [
        helper.make_node("Sin", ["angles"], ["sin"]),
        helper.make_node("Cos", ["angles"], ["cos"]),
        helper.make_node("Tile", ["sin", "twice"], ["sin_full"]),
        helper.make_node("Tile", ["cos", "twice"], ["cos_full"]),
        helper.make_node("Concat", ["sin_full", "cos_full"], ["table"], axis=-1),
        helper.make_node("Tile", ["table", "once"], ["tiled"]),
        helper.make_node("Split", ["tiled"], ["sines", "cosines"], axis=1, num_outputs=2),
        helper.make_node("Split", ["x"], ["first", "second"], axis=3, num_outputs=2),
        helper.make_node("Neg", ["second"], ["negated"]),
        helper.make_node("Concat", ["negated", "first"], ["rotated"], axis=-1),
        helper.make_node("Cast", ["rotated"], ["rotated32"], to=f32),
        helper.make_node("Mul", ["rotated32", "sines"], ["rotated_sin"]),
        helper.make_node("Cast", ["x"], ["x32"], to=f32),
        helper.make_node("Mul", ["x32", "cosines"], ["x_cos"]),
        helper.make_node("Add", ["x_cos", "rotated_sin"], ["sum"]),
        helper.make_node("Cast", ["sum"], ["y"], to=f16),
    ]
    graph = helper.make_graph(
        nodes,
        "rope",
        [
            helper.make_tensor_value_info("x", f16, [1, 2, 5, 8]),
            helper.make_tensor_value_info("angles", f32, [5, 4]),
        ],
        [helper.make_tensor_value_info("y", f16, None)],
        [
            numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "twice"),
            numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "once"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


class OptimizeTests(unittest.TestCase):
    def test_rope_chain_becomes_one_rotary_embedding(self) -> None:
        rng = np.random.default_rng(5)
        feeds = {
            "x": rng.standard_normal((1, 2, 5, 8)).astype(np.float16),
            "angles": rng.uniform(-3, 3, (5, 4)).astype(np.float32),
        }
        original = rope_model()
        run = cast("Run", resolve(ReferenceEvaluator(original), "run"))
        expected = run(None, cast("dict[str, NDArray[np.float32]]", feeds))[0]
        fused = rope_model()
        self.assertEqual(fuse_rotary(fused), 1)
        self.assertEqual(
            [n.op_type for n in fused.graph.node], ["Sin", "Cos", "Cast", "RotaryEmbedding", "Cast"]
        )
        evaluator = ReferenceEvaluator(fused, new_ops=[RotaryEmbedding])
        actual = cast("Run", resolve(evaluator, "run"))(
            None, cast("dict[str, NDArray[np.float32]]", feeds)
        )[0]
        np.testing.assert_array_equal(actual, expected)

    def test_strided_conv_transpose_becomes_pointwise_depth_to_space(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.standard_normal((1, 4, 3, 5)).astype(np.float32)
        for kernel, bias in ((2, True), (4, False)):
            weight = rng.standard_normal((4, 3, kernel, kernel)).astype(np.float32)
            tensors = [numpy_helper.from_array(weight, "w")]
            inputs = ["x", "w"]
            if bias:
                tensors.append(
                    numpy_helper.from_array(rng.standard_normal(3).astype(np.float32), "b")
                )
                inputs.append("b")
            node = helper.make_node(
                "ConvTranspose",
                inputs,
                ["y"],
                kernel_shape=[kernel, kernel],
                strides=[kernel, kernel],
            )
            original = model([node], tensors)
            expected = evaluate(original, x)
            rewritten = model([node], tensors)
            self.assertEqual(rewrite(rewritten), 1)
            self.assertEqual([n.op_type for n in rewritten.graph.node], ["Conv", "DepthToSpace"])
            np.testing.assert_allclose(evaluate(rewritten, x), expected, rtol=1e-5, atol=1e-5)

    def test_overlapping_conv_transpose_is_kept(self) -> None:
        weight = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), "w")
        node = helper.make_node(
            "ConvTranspose", ["x", "w"], ["y"], kernel_shape=[3, 3], strides=[2, 2]
        )
        graph = model([node], [weight])
        self.assertEqual(rewrite(graph), 0)
        self.assertEqual([n.op_type for n in graph.graph.node], ["ConvTranspose"])

    def test_memcpy_nodes_are_removed(self) -> None:
        nodes = [
            helper.make_node("MemcpyFromHost", ["x"], ["device"]),
            helper.make_node("Relu", ["device"], ["y"]),
        ]
        graph = model(nodes, [])
        rewrite(graph)
        self.assertEqual([(n.op_type, list(n.input)) for n in graph.graph.node], [("Relu", ["x"])])


if __name__ == "__main__":
    unittest.main()
