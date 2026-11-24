"""Publish every frame's targets and raw/merged sidecars; acceptance alone retires redundant raw data."""

import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import OpenEXR
from numpy.typing import NDArray

from strata_datagen import files, shotlock
from strata_datagen.exr import Deep, read_deep, read_flat
from strata_datagen.qa.deep import instance_ids
from strata_datagen.render import settings
from strata_dataset import merged, records, roles, segmented
from strata_dataset import samples as contract_samples
from strata_dataset.layers import Layers, reduce_samples

STAGING = ".staging"
REMOVE = "remove them with `strata-datagen remove-samples` while no reader runs, then pack again"


@dataclass(frozen=True)
class Samples:
    """A frame's deep samples, row-major pixels, front to back within a pixel (stable for equal Z)."""

    shape: tuple[int, int]
    offsets: NDArray[np.int64]
    z: NDArray[np.float32]
    alpha: NDArray[np.float32]
    rgb: NDArray[np.float32]
    roles: NDArray[np.float32]
    ids: NDArray[np.int64]

    def arrays(self) -> dict[str, NDArray[np.generic]]:
        return {
            "shape": np.array(self.shape, np.int64),
            "offsets": self.offsets,
            "z": self.z,
            "alpha": self.alpha,
            "rgb": self.rgb,
            "roles": self.roles,
            "ids": self.ids,
        }


@dataclass(frozen=True)
class ShotFrames:
    """The validated shot identity shared by every frame directory."""

    shot_id: str

    @staticmethod
    def load(shot: Path) -> "ShotFrames":
        """The plan of a shot whose records are this contract version; raises ValueError otherwise."""
        manifest = cast("dict[str, object]", json.loads((shot / "manifest.json").read_text()))
        camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
        errors = records.errors(manifest, camera)
        errors = errors or settings.sampling_errors(shot, manifest)
        if errors:
            raise ValueError(f"{shot}: {'; '.join(errors)}")
        return ShotFrames(str(manifest["shot_id"]))

    def sample_dir(self, shot: Path, frame: int) -> Path:
        return shot / "samples" / contract_samples.frame_id(self.shot_id, frame)

    def merged_dir(self, shot: Path, frame: int) -> Path:
        return shot / "merged" / contract_samples.frame_id(self.shot_id, frame)


def samples(deep: Deep) -> Samples:
    """Sample alpha is the float `coverage` channel; the half `A` rounds alpha above 0.99976 to 1.

    Role coverage is the deep role channels, nested within the sample alpha. Samples failing the
    identity check keep their colour, alpha and roles but carry `identity.UNLABELLED`.
    """
    order = segmented.order(deep.channels["Z"], deep.counts)
    alpha = np.ascontiguousarray(np.clip(deep.channels["coverage"][order], 0.0, 1.0), np.float32)
    nested = roles.nested(tuple(deep.channels[role] for role in roles.ROLES), alpha, order=order)
    rgb = cast("NDArray[np.float32]", segmented.gather(tuple(deep.channels[c] for c in "RGB"), order))
    return Samples(
        (deep.height, deep.width),
        deep.offsets.astype(np.int64),
        np.ascontiguousarray(deep.channels["Z"][order], np.float32),
        alpha,
        rgb,
        nested,
        instance_ids(deep, order),
    )


def read_samples(path: Path) -> Samples:
    deep = read_deep(path)
    for channel in ("A", "ZBack"):
        del deep.channels[channel]
    return samples(deep)


def layers(frame: Samples) -> Layers:
    """The contract's K-layer rule with the sky kept: sky clusters keep colour and alpha, depth 0."""
    return reduce_samples(
        frame.z,
        frame.alpha,
        frame.rgb,
        np.diff(frame.offsets),
        frame.shape,
        sky=roles.sky(frame.alpha, frame.roles[:, 0]),
    )


def layer_channels(targets: Layers) -> dict[str, NDArray[np.float32]]:
    out: dict[str, NDArray[np.float32]] = {}
    for k in range(targets.alpha.shape[0]):
        out[f"L{k}.invdepth"] = targets.inverse_depth[k]
        out[f"L{k}.A"] = targets.alpha[k]
        for i, c in enumerate("RGB"):
            out[f"L{k}.{c}"] = targets.rgb[k, i]
    return {name: np.ascontiguousarray(value, np.float32) for name, value in out.items()}


def targets(frame: Samples) -> dict[str, NDArray[np.float32]]:
    """Every channel of a frame's targets: its K layers and the coverage of each role."""
    covered = roles.coverage(frame.alpha, frame.roles, np.diff(frame.offsets), frame.shape)
    return layer_channels(layers(frame)) | covered


def write_layers(values: dict[str, NDArray[np.float32]], out_path: Path) -> None:
    """layers/NNNN.exr: float32 `L<k>.invdepth` (1/m, 0 with sky) and `L<k>.A/R/G/B`, `roles.<role>`."""
    channels = {name: OpenEXR.Channel(value) for name, value in values.items()}
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


def _staging(published: Path) -> Path:
    """A directory no other publish can name, beside `samples/` on the same filesystem."""
    return published.parent.parent / STAGING / f"{published.name}.{os.getpid()}.{secrets.token_hex(4)}"


def _verify_published(published: Path, encoded: dict[str, bytes]) -> None:
    try:
        found = contract_samples.read(published).identity
    except (OSError, ValueError) as error:
        raise ValueError(f"{published} is not a valid published frame: {error}; {REMOVE}") from error
    wanted = contract_samples.identity(encoded)
    if found != wanted:
        raise ValueError(
            f"{published} holds sidecars {found[:12]}, not the {wanted[:12]} packed now;"
            f" published sidecars are immutable: {REMOVE}"
        )


def write_samples(frame: Samples, beauty_path: Path, published: Path) -> None:
    """Publish a frame's sidecars once: stage a whole directory, then rename it to the absent `published`.

    An identical frame already published is left as it is; a different one is refused, never replaced.
    rename(2) cannot replace a non-empty directory, so a published frame survives even a racing writer.
    """
    encoded = contract_samples.encode({**images(beauty_path), **frame.arrays()})
    if published.exists(follow_symlinks=False):
        _verify_published(published, encoded)
        return
    staging = _staging(published)
    try:
        for name, data in encoded.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        published.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(published)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if not published.exists():
            raise
        _verify_published(published, encoded)


def write_merged(frame: Samples, published: Path) -> None:
    """Atomically publish immutable merged deep, including the layer rule's sufficient statistics."""
    runs = merged.merge(
        frame.shape, frame.offsets, frame.z, frame.alpha, frame.rgb, semantic=frame.roles, ids=frame.ids
    )
    arrays = runs.arrays()
    if published.exists():
        found = merged.read(published / "deep.npz").arrays()
        if not all(np.array_equal(value, found[name]) for name, value in arrays.items()):
            raise ValueError(
                f"{published} holds different merged deep; published sidecars are immutable: {REMOVE}"
            )
        return
    staging = _staging(published)
    try:
        staging.mkdir(parents=True)
        np.savez_compressed(staging / "deep.npz", allow_pickle=False, **arrays)
        published.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(published)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _replacement_raw(shot: Path, frame: int) -> None:
    raw = shot / "deep" / f"{frame:04d}.exr"
    replacement = read_samples(raw)
    shape = _shape(shot, frame)
    errors = contract_samples.sample_errors(replacement.arrays(), shape[0] * shape[1])
    if replacement.shape != shape or errors:
        raise ValueError(f"frame {frame}: replacement raw cannot be repacked: {errors}")
    files.sync(raw)
    files.sync(raw.parent)


def remove_samples(shot: Path, frames: range) -> list[int]:
    """Offline: delete the published sidecars of `frames` and abandoned staging; readers must be stopped."""
    plan = ShotFrames.load(shot)
    removed: list[int] = []
    with shotlock.held(shot):
        for frame in frames:
            if not (shot / "deep" / f"{frame:04d}.exr").is_file():
                raise ValueError(f"frame {frame}: cannot remove merged deep without raw; re-render first")
            _replacement_raw(shot, frame)
            removed_frame = False
            for published in (plan.sample_dir(shot, frame), plan.merged_dir(shot, frame)):
                if not published.exists(follow_symlinks=False):
                    continue
                doomed = _staging(published)
                doomed.parent.mkdir(parents=True, exist_ok=True)
                published.rename(doomed)
                removed_frame = True
            if removed_frame:
                removed.append(frame)
        if (shot / STAGING).is_dir():
            shutil.rmtree(shot / STAGING)
    return removed


def _shape(shot: Path, frame: int) -> tuple[int, int]:
    camera = cast("dict[str, object]", json.loads((shot / "camera.json").read_text()))
    records = cast("list[dict[str, int]]", camera["frames"])
    record = next(record for record in records if record["frame"] == frame)
    return record["height"], record["width"]


def pack(shot: Path, frames: range) -> None:
    """Write targets and raw/merged sidecars for every frame; never removes raw deep.

    Role coverage comes from each frame's own retained deep.
    """
    plan = ShotFrames.load(shot)
    with shotlock.held(shot):
        for frame in frames:
            name = f"{frame:04d}.exr"
            raw = shot / "deep" / name
            if not raw.is_file():
                errors = merged.errors(plan.merged_dir(shot, frame) / "deep.npz", _shape(shot, frame))
                if errors or not (shot / "layers" / name).is_file():
                    raise ValueError(f"frame {frame}: missing packed frame or raw deep: {errors}")
                continue
            frame_samples = read_samples(raw)
            write_layers(targets(frame_samples), shot / "layers" / name)
            write_samples(frame_samples, shot / "beauty" / name, plan.sample_dir(shot, frame))
            write_merged(frame_samples, plan.merged_dir(shot, frame))
            del frame_samples
