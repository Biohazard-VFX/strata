"""CPU crop I/O payloads and deterministic device-side hybrid batch finalization."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from strata_training.augment.deep import Shot
from strata_training.augment.pipeline import AugmentConfig, Roles, draw_camera, draw_grain
from strata_training.augment.torch_camera import degrade
from strata_training.augment.torch_compose import TensorShot, recompose
from strata_training.augment.torch_grain import apply_grain
from strata_training.data.raw_tensor import MappedSamples, mapped_samples
from strata_training.torch_api import api
from strata_training.types import Array

if TYPE_CHECKING:
    from strata_training.data.online import OnlineLoader, PreparedBatch


@dataclass(frozen=True)
class TorchJob:
    """Own crop tensors and value-only randomness; safe to pickle across spawn queues."""

    position: int
    shot: TensorShot
    roles: Roles
    donor: TensorShot | None
    donor_roles: Roles | None
    config: AugmentConfig
    seed: int

    def map(
        self, function: Callable[[torch.Tensor], torch.Tensor], pin: bool = False
    ) -> "TorchJob":
        def shot(value: TensorShot) -> TensorShot:
            deep = value.deep
            if isinstance(deep, MappedSamples):
                deep = deep if pin else deep.transfer(function)
            elif deep is not None:
                deep = replace(
                    deep,
                    **{
                        k: function(getattr(deep, k))
                        for k in ("counts", "z", "alpha", "rgb", "ids")
                    },
                )
            return TensorShot(deep, function(value.noisy), function(value.denoised))

        return replace(self, shot=shot(self.shot), donor=shot(self.donor) if self.donor else None)


def _owned(value: Array) -> torch.Tensor:
    return api.from_numpy(np.array(value, copy=True))


def _read_shot(shot: Shot, top: int, left: int, size: int, raw: bool, root: Path) -> TensorShot:
    deep = mapped_samples(root, top, left, size, size) if raw else None
    window = (slice(None), slice(top, top + size), slice(left, left + size))
    return TensorShot(deep, _owned(shot.noisy[window]), _owned(shot.denoised[window]))


def prepare_raw(
    loader: "OnlineLoader",
    selections: list[tuple[int, int, int]],
    rng: np.random.Generator,
    crop: int,
    halo: int,
) -> "PreparedBatch":
    from strata_training.data.online import PreparedBatch

    size = crop + 2 * halo
    clean: list[int] = []
    live: list[int] = []
    jobs: list[TorchJob] = []
    images: list[torch.Tensor] = []
    targets: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("inverse_depth", "alpha", "rgb", "beauty", "mixed")
    }
    scales: list[float] = []
    for position, (i, top, left) in enumerate(selections):
        frame = loader.frames[i]
        window = (slice(top, top + size), slice(left, left + size))
        layers = {
            key: _owned(frame.arrays[key][..., *window])
            for key in ("inverse_depth", "alpha", "rgb")
        }
        image = _owned(loader.shot(i).denoised[:, *window])
        beauty = _owned(frame.arrays.get("target_beauty", frame.arrays["beauty"])[:, *window])
        if rng.random() < loader.clean_fraction:
            clean.append(position)
            scale = frame.scale
        else:
            live.append(position)
            flags = {
                key: float(rng.random() < getattr(loader.augmentation, key))
                for key in (
                    "scene_grade",
                    "instance_grade",
                    "recompose",
                    "occluders",
                    "grain",
                    "camera",
                )
            }
            config = replace(loader.augmentation, **flags)
            seed = int(rng.integers(0, 2**63))
            raw = bool(config.instance_grade or config.recompose or config.occluders)
            shot = _read_shot(
                loader.shot(i), top, left, size, raw, loader.samples / frame.entry["id"] / "deep"
            )
            donor: TensorShot | None = None
            donor_roles: Roles | None = None
            if config.recompose:
                donor_i = int(rng.integers(len(loader.frames)))
                y, x = loader.frames[donor_i].crop(rng, size)
                donor = _read_shot(
                    loader.shot(donor_i),
                    y,
                    x,
                    size,
                    True,
                    loader.samples / loader.frames[donor_i].entry["id"] / "deep",
                )
                donor_roles = loader.roles[donor_i]
            jobs.append(TorchJob(position, shot, loader.roles[i], donor, donor_roles, config, seed))
            scale = frame.scale * size / frame.base.shape[1]
        images.append(image)
        scales.append(scale)
        values = {
            **layers,
            "beauty": beauty,
            "mixed": (layers["alpha"][0] > 0.03) & (layers["alpha"][0] < 0.97),
        }
        for key in targets:
            targets[key].append(values[key].float())
    return PreparedBatch(
        torch.stack(images),
        {k: torch.stack(v) for k, v in targets.items()},
        clean,
        live,
        selections,
        torch.tensor(scales)[:, None, None, None],
        None,
        jobs=jobs,
        device_math=True,
        crop=crop,
        halo=halo,
    )


def normalize(images: torch.Tensor) -> torch.Tensor:
    from strata_training.data.cache import DISPLAY_EXPOSURE, DISPLAY_MATRIX

    matrix = torch.tensor(DISPLAY_MATRIX, device=images.device)
    linear = (torch.einsum("ij,bjhw->bihw", matrix, images.double()) * DISPLAY_EXPOSURE).clamp(0, 1)
    srgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    display = ((srgb * 255 + 0.5).floor() / 255).float()
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device)[None, :, None, None]
    return (display - mean) / std


def finalize(prepared: "PreparedBatch", crop: int, halo: int) -> "PreparedBatch":
    """Consume owned raw payloads on their device; return training-ready native tensors."""
    prepared = replace(
        prepared,
        images=prepared.images.clone(),
        target={k: v.clone() for k, v in prepared.target.items()},
    )
    for job in prepared.jobs:
        rng = np.random.default_rng(job.seed)
        record: dict[str, Any] = {}
        position = job.position
        targets = {
            k: v[position]
            for k, v in prepared.target.items()
            if k in ("inverse_depth", "alpha", "rgb")
        }
        image, targets = recompose(
            job.shot, job.roles, targets, job.donor, job.donor_roles, rng, job.config, record
        )
        alpha, rgb = targets["alpha"], targets["rgb"]
        through = torch.cat((torch.ones_like(alpha[:1]), (1 - alpha).cumprod(0)[:-1]))
        beauty = (through[:, None] * rgb).sum(0)
        values = {
            **targets,
            "beauty": beauty,
            "mixed": ((alpha[0] > 0.03) & (alpha[0] < 0.97)).float(),
        }
        for key, value in values.items():
            prepared.target[key][position] = value
        grain = draw_grain(rng, job.config)
        if grain is not None:
            image = apply_grain(image, grain)
        camera = draw_camera(rng, job.config)
        if camera is not None:
            image = degrade(image, camera)
        prepared.images[position] = image
    prepared.normalized = normalize(prepared.images[prepared.live]) if prepared.live else None
    prepared.target = {
        k: v[..., halo : halo + crop, halo : halo + crop] for k, v in prepared.target.items()
    }
    prepared.jobs = []
    prepared.device_math = False
    return prepared
