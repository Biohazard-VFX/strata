"""After rendering: sky-inclusive K-layer targets for every frame, raw-sample sidecars on the deep stride.

Packing only adds files. Raw deep is QA evidence; only `qa.checks.prune` removes it, after the frame
has passed raw QA.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_contract import samples as contract_samples
from strata_contract.layers import Layers, reduce_samples
from strata_datagen.exr import Deep, read_deep, read_flat
from strata_datagen.generators.lighting import SKY_ID
from strata_datagen.qa.deep import instance_ids

DEEP_STRIDE = 24


@dataclass(frozen=True)
class Samples:
    """A frame's deep samples, row-major pixels, front to back within a pixel (stable for equal Z)."""

    shape: tuple[int, int]
    offsets: NDArray[np.int64]
    z: NDArray[np.float32]
    alpha: NDArray[np.float32]
    rgb: NDArray[np.float32]
    ids: NDArray[np.int64]

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "shape": np.array(self.shape, np.int64),
            "offsets": self.offsets,
            "z": self.z,
            "alpha": self.alpha,
            "rgb": self.rgb,
            "ids": self.ids,
        }


@dataclass(frozen=True)
class ShotFrames:
    """The frame plan a manifest declares: frames on the deep stride are recomposition frames."""

    shot_id: str
    first: int
    stride: int

    @staticmethod
    def load(shot: Path) -> "ShotFrames":
        manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
        render = cast("dict[str, object]", manifest["render"])
        frames = cast("list[int]", render["frames"])
        return ShotFrames(str(manifest["shot_id"]), frames[0], cast("int", render["deep_stride"]))

    def recomposition(self, frame: int) -> bool:
        return (frame - self.first) % self.stride == 0

    def sample_dir(self, shot: Path, frame: int) -> Path:
        return shot / "samples" / contract_samples.frame_id(self.shot_id, frame)


def samples(deep: Deep) -> Samples:
    """Sample alpha is the float `coverage` channel; the half `A` rounds alpha above 0.99976 to 1."""
    pixel = deep.pixel_index()
    order = np.lexsort((deep.channels["Z"], pixel))
    rgb = np.stack([deep.channels[c] for c in "RGB"], axis=-1)
    return Samples(
        (deep.height, deep.width),
        deep.offsets.astype(np.int64),
        np.ascontiguousarray(deep.channels["Z"][order], np.float32),
        np.ascontiguousarray(np.clip(deep.channels["coverage"][order], 0.0, 1.0), np.float32),
        np.ascontiguousarray(rgb[order], np.float32),
        np.ascontiguousarray(instance_ids(deep)[order], np.int64),
    )


def layers(frame: Samples) -> Layers:
    """The contract's K-layer rule with the sky kept: sky clusters keep colour and alpha, depth 0."""
    return reduce_samples(
        frame.z.astype(np.float64),
        frame.alpha.astype(np.float64),
        frame.rgb.astype(np.float64),
        np.diff(frame.offsets),
        frame.shape,
        sky=frame.ids == SKY_ID,
    )


def layer_channels(targets: Layers) -> dict[str, NDArray[np.float32]]:
    out: dict[str, NDArray[np.float32]] = {}
    for k in range(targets.alpha.shape[0]):
        out[f"L{k}.invdepth"] = targets.inverse_depth[k]
        out[f"L{k}.A"] = targets.alpha[k]
        for i, c in enumerate("RGB"):
            out[f"L{k}.{c}"] = targets.rgb[k, i]
    return {name: np.ascontiguousarray(value, np.float32) for name, value in out.items()}


def write_layers(targets: Layers, out_path: Path) -> None:
    """layers/NNNN.exr: per layer k, float32 `L<k>.invdepth` (1/m, 0 with sky) and `L<k>.A/R/G/B`."""
    channels = {name: OpenEXR.Channel(value) for name, value in layer_channels(targets).items()}
    header: dict[str, object] = {"compression": OpenEXR.ZIPS_COMPRESSION, "type": OpenEXR.scanlineimage}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(".tmp.exr")
    OpenEXR.File(header, channels).write(str(temporary))
    temporary.replace(out_path)


def images(beauty_path: Path) -> dict[str, NDArray[np.float32]]:
    beauty = read_flat(beauty_path)
    return {
        "noisy": np.stack([beauty[c] for c in "RGB"]).astype(np.float32),
        "denoised": np.stack([beauty[f"denoised.{c}"] for c in "RGB"]).astype(np.float32),
    }


def write_samples(frame: Samples, beauty_path: Path, directory: Path) -> None:
    """Publish a frame's sidecar directory atomically: written beside it, then renamed into place."""
    temporary = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    (temporary / "deep").mkdir(parents=True)
    for name, value in images(beauty_path).items():
        np.save(temporary / f"{name}.npy", np.ascontiguousarray(value), allow_pickle=False)
    for name, value in frame.arrays().items():
        np.save(temporary / "deep" / f"{name}.npy", np.ascontiguousarray(value), allow_pickle=False)
    if directory.exists():
        stale = directory.with_name(directory.name + ".old")
        directory.replace(stale)
        temporary.replace(directory)
        shutil.rmtree(stale)
    else:
        temporary.replace(directory)


def pack(shot: Path, frames: range) -> None:
    """Write targets for `frames` and sidecars for their recomposition frames; deletes nothing."""
    plan = ShotFrames.load(shot)
    for frame in frames:
        name = f"{frame:04d}.exr"
        frame_samples = samples(read_deep(shot / "deep" / name))
        write_layers(layers(frame_samples), shot / "layers" / name)
        if plan.recomposition(frame):
            write_samples(frame_samples, shot / "beauty" / name, plan.sample_dir(shot, frame))
