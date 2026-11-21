"""Bake ONNX Runtime's portable graph rewrites and exact layout rewrites into an export."""

from __future__ import annotations

import argparse
import importlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import helper, numpy_helper

from ._external import resolve
from .fold_constants import SaveModel


class Session(Protocol):
    def __call__(self, path: str, options: object, *, providers: list[object]) -> object: ...


class Options(Protocol):
    graph_optimization_level: object
    optimized_model_filepath: str

    def add_session_config_entry(self, key: str, value: str) -> None: ...


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    get = cast("Callable[[onnx.AttributeProto], object]", resolve(helper, "get_attribute_value"))
    return {attribute.name: get(attribute) for attribute in node.attribute}


def _pointwise(node: onnx.NodeProto, tensors: dict[str, onnx.TensorProto]) -> bool:
    if node.op_type != "ConvTranspose" or node.input[1] not in tensors:
        return False
    attrs = _attributes(node)
    kernel = cast("list[int]", attrs.get("kernel_shape", []))
    return (
        len(kernel) == 2
        and kernel == cast("list[int]", attrs.get("strides", [1, 1]))
        and cast("int", attrs.get("group", 1)) == 1
        and not any(cast("list[int]", attrs.get("pads", [0])))
        and not any(cast("list[int]", attrs.get("output_padding", [0])))
        and cast("list[int]", attrs.get("dilations", [1, 1])) == [1, 1]
    )


def _array(tensor: onnx.TensorProto) -> NDArray[np.generic]:
    return cast("NDArray[np.generic]", numpy_helper.to_array(tensor))


def _tensor(array: NDArray[np.generic], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def _depth_to_space(
    node: onnx.NodeProto, tensors: dict[str, onnx.TensorProto]
) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    weight = _array(tensors[node.input[1]])
    channels_in, channels_out, kernel = weight.shape[0], weight.shape[1], weight.shape[2]
    packed = weight.transpose(1, 2, 3, 0).reshape(channels_out * kernel * kernel, channels_in, 1, 1)
    added = [_tensor(np.ascontiguousarray(packed), node.input[1] + "_pointwise")]
    inputs = [node.input[0], added[0].name]
    if len(node.input) > 2 and node.input[2]:
        bias = np.repeat(_array(tensors[node.input[2]]), kernel * kernel)
        added.append(_tensor(bias, node.input[2] + "_pointwise"))
        inputs.append(added[1].name)
    packed_name = node.output[0] + "_packed"
    nodes = [
        helper.make_node(
            "Conv", inputs, [packed_name], name=node.name + "_pointwise", kernel_shape=[1, 1]
        ),
        helper.make_node(
            "DepthToSpace",
            [packed_name],
            [node.output[0]],
            name=node.name + "_depth_to_space",
            blocksize=kernel,
            mode="CRD",
        ),
    ]
    return nodes, added


class _Graph:
    def __init__(self, nodes: list[onnx.NodeProto], tensors: dict[str, onnx.TensorProto]) -> None:
        self.tensors = tensors
        self.producer = {name: node for node in nodes for name in node.output}
        self.consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in nodes:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)

    def only(self, name: str, op: str) -> onnx.NodeProto | None:
        nodes = self.consumers.get(name, [])
        return nodes[0] if len(nodes) == 1 and nodes[0].op_type == op else None

    def made(self, name: str, op: str) -> onnx.NodeProto | None:
        node = self.producer.get(name)
        return node if node is not None and node.op_type == op else None

    def table(self, name: str) -> onnx.NodeProto | None:
        node = self.producer.get(name)
        while node is not None and node.op_type == "Tile" and node.input[1] in self.tensors:
            if not bool(np.all(_array(self.tensors[node.input[1]]) == 1)):
                return None
            node = self.producer.get(node.input[0])
        return node if node is not None and node.op_type == "Concat" else None


def _other(node: onnx.NodeProto, name: str) -> str:
    return node.input[1] if node.input[0] == name else node.input[0]


def _rotary(
    graph: _Graph, neg: onnx.NodeProto
) -> tuple[onnx.NodeProto, list[onnx.NodeProto]] | None:
    split = graph.made(neg.input[0], "Split")
    if split is None or list(split.output)[1:] != [neg.input[0]]:
        return None
    concat = graph.only(neg.output[0], "Concat")
    if concat is None or list(concat.input) != [neg.output[0], split.output[0]]:
        return None
    widen = graph.only(concat.output[0], "Cast")
    product = graph.only(widen.output[0], "Mul") if widen is not None else None
    total = graph.only(product.output[0], "Add") if product is not None else None
    if widen is None or product is None or total is None:
        return None
    direct = graph.made(_other(total, product.output[0]), "Mul")
    if direct is None:
        return None
    for name in direct.input:
        cast = graph.made(name, "Cast")
        if cast is None or cast.input[0] != split.input[0]:
            continue
        sines, cosines = _other(product, widen.output[0]), _other(direct, name)
        halves = graph.made(sines, "Split")
        table = graph.table(halves.input[0]) if halves is not None else None
        if halves is None or table is None or graph.producer.get(cosines) is not halves:
            return None
        tiles = [graph.made(source, "Tile") for source in table.input]
        sin_tile = tiles[list(halves.output).index(sines)]
        cos_tile = tiles[list(halves.output).index(cosines)]
        if sin_tile is None or cos_tile is None:
            return None
        node = helper.make_node(
            "RotaryEmbedding",
            [cast.output[0], ROTARY_POSITION, cos_tile.input[0], sin_tile.input[0]],
            [total.output[0]],
            name=total.name + "_rotary",
            domain="com.microsoft",
            interleaved=0,
        )
        return node, [neg, concat, widen, product, total, direct]
    return None


ROTARY_POSITION = "strata_rotary_position"


def fuse_rotary(model: onnx.ModelProto) -> int:
    """Replace float32 rotate-half RoPE chains with one com.microsoft RotaryEmbedding each."""
    nodes = list(model.graph.node)
    graph = _Graph(nodes, {tensor.name: tensor for tensor in model.graph.initializer})
    replaced: dict[int, onnx.NodeProto] = {}
    removed: set[int] = set()
    for neg in nodes:
        found = _rotary(graph, neg) if neg.op_type == "Neg" else None
        if found is None:
            continue
        node, old = found
        replaced[id(old[4])] = node
        removed.update(id(item) for item in old)
    if not replaced:
        return 0
    kept = [
        replaced.get(id(node), node)
        for node in nodes
        if id(node) not in removed or id(node) in replaced
    ]
    outputs = {value.name for value in model.graph.output}
    while True:
        used = {name for node in kept for name in node.input} | outputs
        alive = [node for node in kept if any(name in used for name in node.output if name)]
        if len(alive) == len(kept):
            break
        kept = alive
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.initializer.append(_tensor(np.zeros(1, dtype=np.int64), ROTARY_POSITION))
    if all(opset.domain != "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return len(replaced)


def rewrite(model: onnx.ModelProto) -> int:
    """Drop runtime memcpy nodes and express stride==kernel ConvTranspose as Conv+DepthToSpace."""
    graph = model.graph
    tensors = {tensor.name: tensor for tensor in graph.initializer}
    renamed: dict[str, str] = {}
    nodes: list[onnx.NodeProto] = []
    added: list[onnx.TensorProto] = []
    rewritten = 0
    for node in graph.node:
        if node.op_type in ("MemcpyFromHost", "MemcpyToHost"):
            renamed[node.output[0]] = node.input[0]
        elif _pointwise(node, tensors):
            replacement, weights = _depth_to_space(node, tensors)
            nodes.extend(replacement)
            added.extend(weights)
            rewritten += 1
        else:
            nodes.append(node)
    for node in nodes:
        for index, name in enumerate(node.input):
            node.input[index] = renamed.get(name, name)
    del graph.node[:]
    graph.node.extend(nodes)
    used = {name for node in nodes for name in node.input}
    kept = [tensor for tensor in [*graph.initializer, *added] if tensor.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return rewritten


def optimize(path: Path) -> None:
    ort = importlib.import_module("onnxruntime")
    load = cast("Callable[[str], onnx.ModelProto]", resolve(onnx, "load"))
    save = cast("SaveModel", resolve(onnx, "save_model"))
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        options = cast("Options", cast("Callable[[], object]", resolve(ort, "SessionOptions"))())
        levels = resolve(ort, "GraphOptimizationLevel")
        options.graph_optimization_level = resolve(levels, "ORT_ENABLE_BASIC")
        options.optimized_model_filepath = str(staged)
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name", path.name + ".basic"
        )
        options.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes", "1024"
        )
        session = cast("Session", resolve(ort, "InferenceSession"))
        session(str(path), options, providers=[("CUDAExecutionProvider", {"use_tf32": "0"})])
        model = load(str(staged))
        rewritten = rewrite(model)
        fused = fuse_rotary(model)
        result = Path(directory) / "result"
        result.mkdir()
        save(
            model,
            str(result / path.name),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".weights",
            size_threshold=1024,
        )
        (result / (path.name + ".weights")).replace(path.with_name(path.name + ".weights"))
        (result / path.name).replace(path)
    print(f"optimized {path}: {rewritten} ConvTranspose, {fused} RoPE fused", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    for model in cast("list[Path]", parser.parse_args().models):
        optimize(model)


if __name__ == "__main__":
    main()
