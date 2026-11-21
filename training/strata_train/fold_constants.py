"""Materialize constant precision casts before distributing model weights."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import onnx
from onnx import numpy_helper

from ._external import resolve


class SaveModel(Protocol):
    def __call__(
        self,
        model: onnx.ModelProto,
        path: str,
        *,
        save_as_external_data: bool,
        all_tensors_to_one_file: bool,
        location: str,
        size_threshold: int,
    ) -> None: ...


def fold(path: Path) -> None:
    load = cast("Callable[[str], onnx.ModelProto]", resolve(onnx, "load"))
    save = cast("SaveModel", resolve(onnx, "save_model"))
    check = cast("Callable[[str], None]", resolve(onnx.checker, "check_model"))
    model = load(str(path))
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    nodes: list[onnx.NodeProto] = []
    folded = 0
    for node in model.graph.node:
        if node.op_type == "Cast" and node.input[0] in initializers:
            dtype = next(attr.i for attr in node.attribute if attr.name == "to")
            array = numpy_helper.to_array(initializers[node.input[0]])
            converted = array.astype(onnx.helper.tensor_dtype_to_np_dtype(dtype))
            initializers[node.output[0]] = numpy_helper.from_array(converted, node.output[0])
            folded += 1
        else:
            nodes.append(node)
    used = {name for node in nodes for name in node.input} | {
        value.name for value in model.graph.output
    }
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensor for name, tensor in initializers.items() if name in used)
    data = path.with_name(path.name + ".weights")
    data.unlink(missing_ok=True)
    temporary = path.with_name(path.name + ".folded")
    save(
        model,
        str(temporary),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data.name,
        size_threshold=1024,
    )
    check(str(temporary))
    temporary.replace(path)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    print(f"folded {folded} constant Casts: {path}", flush=True)
