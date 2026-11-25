import argparse
import hashlib
import os
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import torch
from onnxscript import FLOAT, INT64
from onnxscript import opset23 as op
from torch import nn

from strata_training.eval.predict import load_model
from strata_training.io import atomic_path
from strata_training.model.checkpoint import save_checkpoint
from strata_training.model.head import COVERAGE_OBJECTIVE, DeepHead

onnx_io: Any = import_module("onnx")
torch_api: Any = import_module("torch")


def reduce_product(inp: FLOAT, dims: list[int], output_dtype: int = 1) -> FLOAT:
    del output_dtype
    axes = cast(INT64, op.Constant(value_ints=dims))
    return cast(FLOAT, op.ReduceProd(inp, axes, keepdims=0))


def cumulative_sum(inp: FLOAT, dim: int, dtype: int = 1) -> FLOAT:
    del dtype
    axis = cast(INT64, op.Constant(value_int=dim))
    return cast(FLOAT, op.CumSum(inp, axis))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_backbone(backbone: Path, output: Path) -> None:
    graph: onnx.ModelProto = onnx_io.load(backbone, load_external_data=False)
    copied: set[Path] = set()
    producers = {value: node for node in graph.graph.node for value in node.output}
    value = graph.graph.output[0].name
    for expected in ("Relu", "Conv", "Cast", "Relu", "Conv"):
        node = producers[value]
        if node.op_type != expected:
            raise ValueError("Backbone inverse head does not match the frozen DP2 export")
        value = node.input[0]
    info = next(info for info in graph.graph.value_info if info.name == value)
    if info.type.tensor_type.shape.dim[1].dim_value != 128:
        raise ValueError("Runtime requires the H+ 128-channel feature tap")
    graph.graph.node.append(
        onnx.helper.make_node("Cast", [value], ["strata_features"], to=onnx.TensorProto.FLOAT)
    )
    shape = info.type.tensor_type.shape.dim
    graph.graph.output.append(
        onnx.helper.make_tensor_value_info(
            "strata_features",
            onnx.TensorProto.FLOAT,
            [dim.dim_value or dim.dim_param for dim in shape],
        )
    )
    for tensor in graph.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        data = {entry.key: entry.value for entry in tensor.external_data}
        source = (backbone.parent / data["location"]).resolve()
        target = output.parent / source.name
        if target.exists():
            if source not in copied and file_sha256(source) != file_sha256(target):
                raise ValueError("External weight target already exists")
        else:
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        copied.add(source)
        next(entry for entry in tensor.external_data if entry.key == "location").value = source.name
    with atomic_path(output) as temporary:
        onnx_io.save(graph, temporary)


class RuntimeHead(nn.Module):
    def __init__(self, head: DeepHead) -> None:
        super().__init__()
        self.head = head

    def forward(
        self, features: torch.Tensor, image: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.head(features, image, base)
        return prediction["inverse_depth"], prediction["alpha"]


def smoke_checkpoint(path: Path) -> None:
    if "plumbing-only" not in path.name:
        raise ValueError("Smoke checkpoint filename must contain plumbing-only")
    with torch_api.random.fork_rng(devices=[]):
        torch_api.manual_seed(0)
        head = DeepHead(128, nn.Identity(), 4)
        with torch.no_grad():
            head.output.weight.normal_(0, 0.1)
            cast(torch.Tensor, head.output.bias).normal_(0, 0.01)
        optimizer = torch.optim.AdamW(head.parameters())
        save_checkpoint(
            path,
            head,
            optimizer,
            {
                "mode": "head",
                "preset": "depthpro2-h",
                "layers": 4,
                "objective": COVERAGE_OBJECTIVE,
                "purpose": "plumbing-only",
                "trained": False,
            },
            np.random.default_rng(0),
            0,
            0.0,
        )


def export_head(checkpoint: Path, output: Path, backbone: Path) -> None:
    model, config = load_model(checkpoint, "cpu")
    if (
        not isinstance(model, DeepHead)
        or config["layers"] != 4
        or config["preset"] != "depthpro2-h"
    ):
        raise ValueError("Runtime requires a four-layer frozen-feature head")
    if config.get("purpose") == "plumbing-only" and "plumbing-only" not in output.name:
        raise ValueError("Smoke ONNX filename must contain plumbing-only")
    companion = output.with_suffix(".backbone.onnx")
    for target in (output, companion):
        if target.resolve() in {checkpoint.resolve(), backbone.resolve()}:
            raise ValueError("Export must not replace a source artifact")
    feature_backbone(backbone, companion)
    channels = model.stem.in_channels - 4
    example = (
        torch.zeros(1, channels, 32, 48),
        torch.zeros(1, 3, 32, 48),
        torch.ones(1, 1, 32, 48),
    )
    height = torch.export.Dim("height", min=1)
    width = torch.export.Dim("width", min=1)
    shapes: dict[str, dict[int, Any]] = {
        name: {2: height, 3: width} for name in ("features", "image", "base")
    }
    with atomic_path(output) as temporary:
        program: Any = torch_api.onnx.export(
            RuntimeHead(model).eval(),
            example,
            dynamo=True,
            opset_version=23,
            input_names=["features", "image", "base"],
            output_names=["inverse_depth", "alpha"],
            dynamic_shapes=shapes,
            external_data=False,
            custom_translation_table={
                torch.ops.prims.prod.default: reduce_product,
                torch.ops.aten.cumsum.default: cumulative_sum,
            },
        )
        program.save(str(temporary), external_data=False)
        graph: onnx.ModelProto = onnx_io.load(temporary)
        onnx.helper.set_model_props(
            graph,
            {
                "strata.objective": COVERAGE_OBJECTIVE,
                "strata.layers": "4",
                "strata.purpose": str(config.get("purpose", "trained-head")),
                "strata.checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "strata.colour": "plate-premultiplied-conditional-alpha",
                "strata.backbone_sha256": file_sha256(backbone),
                "strata.feature_backbone_sha256": file_sha256(companion),
            },
        )
        onnx_io.checker.check_model(graph, full_check=True)
        onnx_io.save(graph, temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--create-plumbing-only", action="store_true")
    args = parser.parse_args()
    if args.create_plumbing_only:
        if args.checkpoint.exists():
            raise ValueError("Refusing to replace an existing checkpoint")
        smoke_checkpoint(args.checkpoint)
    export_head(args.checkpoint, args.out, args.backbone)


if __name__ == "__main__":
    main()
