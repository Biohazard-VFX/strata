"""Export native-resolution Depth Pro 2 H+ with memory-efficient ONNX Attention."""

from __future__ import annotations

import argparse
import math
import types
from pathlib import Path
from typing import Protocol, cast

import torch
from depth_pro_2 import DepthPro2, create_model_and_transforms
from depth_pro_2.network.fov import FOVEstimator
from onnxscript import FLOAT
from onnxscript import opset23 as op
from strata_contract import write_manifest
from torch import Tensor, nn

from ._external import resolve
from ._precision import FloatProjection
from .fold_constants import fold
from .optimize import optimize


class Exported(Protocol):
    def save(self, destination: str, *, external_data: bool) -> None: ...


class Exporter(Protocol):
    def __call__(
        self,
        model: nn.Module,
        args: tuple[Tensor],
        *,
        dynamo: bool,
        opset_version: int,
        input_names: list[str],
        output_names: list[str],
        dynamic_shapes: dict[str, dict[int, object]],
        external_data: bool,
        custom_translation_table: dict[object, object],
        optimize: bool,
    ) -> Exported: ...


def attention(
    query: FLOAT,
    key: FLOAT,
    value: FLOAT,
    attn_mask: FLOAT | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> FLOAT:
    if attn_mask is not None or dropout_p or is_causal or enable_gqa:
        msg = "H+ export expects unmasked, noncausal, equal-head attention"
        raise ValueError(msg)
    outputs = cast(
        "tuple[FLOAT, FLOAT, FLOAT, FLOAT]", op.Attention(query, key, value, scale=scale)
    )
    return outputs[0]


def dynamic_rope(
    self: nn.Module,
    shape: tuple[int, int],
    dtype: torch.dtype | None = None,
) -> Tensor:
    # timm's float(max(shape)) freezes the trace; keep the denominator tensor-valued.
    periods = cast("Tensor", resolve(self, "periods"))
    h, w = shape
    denom = torch.scalar_tensor(h, dtype=torch.float32, device=periods.device).maximum(
        torch.scalar_tensor(w, dtype=torch.float32, device=periods.device)
    )
    y = torch.arange(h, device=periods.device, dtype=torch.float32) + 0.5
    x = torch.arange(w, device=periods.device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(y / denom * 2 - 1, x / denom * 2 - 1, indexing="ij")
    coords = torch.stack((yy, xx), -1).flatten(0, 1)
    angles = (2 * math.pi * coords[:, :, None] / periods[None, None, :]).flatten(1)
    if not cast("bool", resolve(self, "rotate_half")):
        msg = "H+ export requires half-layout rotary embeddings"
        raise ValueError(msg)
    result = torch.cat((angles.sin().repeat(1, 2), angles.cos().repeat(1, 2)), -1)
    return result if dtype is None else result.to(dtype)


class Depth(nn.Module):
    def __init__(self, model: DepthPro2) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        with torch.autocast(
            image.device.type,
            dtype=self.model.precision,
            enabled=self.model.precision != torch.float32,
        ):
            return self.model.forward(image).float()


class Focal(nn.Module):
    def __init__(self, fov: FOVEstimator) -> None:
        super().__init__()
        self.fov = fov

    def forward(self, image: Tensor) -> Tensor:
        with torch.autocast(
            image.device.type, dtype=self.fov.precision, enabled=self.fov.precision != torch.float32
        ):
            tokens, size = self.fov.encode(image)
        logits = self.fov.head.forward(tokens, size)
        return 1.0 + torch.sigmoid(logits) * 178.0


def export_network(net: nn.Module, path: Path, output: str) -> None:
    net = net.cuda().eval()
    example = torch.zeros((1, 3, 256, 384), device="cuda")
    shapes: dict[str, dict[int, object]] = {
        "image": {
            2: 32 * torch.export.Dim("h", min=1, max=136),
            3: 32 * torch.export.Dim("w", min=1, max=240),
        },
    }
    exporter = cast("Exporter", resolve(torch.onnx, "export"))
    program = exporter(
        net,
        (example,),
        dynamo=True,
        opset_version=23,
        input_names=["image"],
        output_names=[output],
        dynamic_shapes=shapes,
        external_data=True,
        custom_translation_table={torch.ops.aten.scaled_dot_product_attention.default: attention},
        optimize=True,
    )
    program.save(str(path), external_data=True)
    del program
    net.cpu()
    torch.cuda.empty_cache()
    fold(path)
    optimize(path)
    print(f"exported {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    out: Path = args.out
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    torch.set_num_threads(4)
    model, _ = create_model_and_transforms("depthpro2-h", device="cpu", precision=dtype)
    fov = model.fov
    if fov is None:
        msg = "H+ focal estimator missing"
        raise RuntimeError(msg)
    attribute = "get_embed"
    setattr(model.encoder.rope, attribute, types.MethodType(dynamic_rope, model.encoder.rope))
    setattr(fov.trunk.rope, attribute, types.MethodType(dynamic_rope, fov.trunk.rope))
    if dtype == torch.float16:
        model.head[-2] = FloatProjection(model.head[-2])
    model.fov = None
    out.parent.mkdir(parents=True, exist_ok=True)
    export_network(Depth(model), out, "network_inverse_depth")
    export_network(Focal(fov), out.with_suffix(".fov.onnx"), "fov_deg")
    write_manifest(out)


if __name__ == "__main__":
    main()
