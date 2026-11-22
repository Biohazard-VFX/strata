"""After rendering: write compact K-layer targets for every frame and keep raw deep only on a stride."""

from pathlib import Path

import numpy as np
import OpenEXR

from strata_contract.layers import reduce_samples
from strata_datagen.exr import read_deep
from strata_datagen.generators.lighting import SKY_ID
from strata_datagen.qa.deep import instance_ids

DEEP_STRIDE = 24


def keeps_deep(frame: int, first: int, stride: int) -> bool:
    return (frame - first) % stride == 0


def write_layers(deep_path: Path, out_path: Path) -> None:
    """layers/NNNN.exr: per layer k, `L<k>.invdepth` (float, 1/m) and `L<k>.A/R/G/B` (half).

    Layers describe the scene in front of the sky: backdrop samples (id 0) are left out.
    """
    deep = read_deep(deep_path)
    scene = instance_ids(deep) != SKY_ID
    counts = np.bincount(deep.pixel_index()[scene], minlength=deep.counts.size)
    rgb = np.stack([deep.channels[c][scene] for c in "RGB"], axis=-1).astype(np.float64)
    layers = reduce_samples(
        deep.channels["Z"][scene].astype(np.float64),
        deep.channels["A"][scene].astype(np.float64),
        rgb,
        counts,
        (deep.height, deep.width),
    )
    channels: dict[str, OpenEXR.Channel] = {}
    for k in range(layers.alpha.shape[0]):
        channels[f"L{k}.invdepth"] = OpenEXR.Channel(np.ascontiguousarray(layers.inverse_depth[k]))
        channels[f"L{k}.A"] = OpenEXR.Channel(np.ascontiguousarray(layers.alpha[k].astype(np.float16)))
        for i, c in enumerate("RGB"):
            channels[f"L{k}.{c}"] = OpenEXR.Channel(np.ascontiguousarray(layers.rgb[k, i].astype(np.float16)))
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.scanlineimage}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp.exr")
    OpenEXR.File(header, channels).write(str(temporary))
    temporary.replace(out_path)


def pack(shot: Path, frames: range, first: int, stride: int) -> list[int]:
    """Write layers for `frames`; delete their raw deep unless on the stride. Returns frames kept."""
    kept: list[int] = []
    for frame in frames:
        deep_path = shot / "deep" / f"{frame:04d}.exr"
        write_layers(deep_path, shot / "layers" / f"{frame:04d}.exr")
        if keeps_deep(frame, first, stride):
            kept.append(frame)
        else:
            deep_path.unlink()
    return kept
