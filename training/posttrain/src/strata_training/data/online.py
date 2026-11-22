"""Native crop loader; clean samples reuse features, changed samples encode live."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from strata_training.augment.deep import (
    Shot,
    load_samples,
    read_samples,
    save_samples,
)
from strata_training.augment.pipeline import (
    AugmentConfig,
    Donor,
    Roles,
    augment_sample,
    photometric,
    roles,
)
from strata_training.data.cache import display_rgb
from strata_training.data.dataset import Frame, input_batch
from strata_training.data.exr import read_rgb
from strata_training.data.policy import require_supervision
from strata_training.data.targets import composite
from strata_training.data.torch_online import TorchJob
from strata_training.io import save_array
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.live import FrozenEncoder
from strata_training.types import Array, Layers


def image_digest(image: Array) -> str:
    """Hash borrowed image values without mutation; safe for immutable shared inputs."""
    return hashlib.sha256(np.asarray(image, dtype=np.float32).tobytes()).hexdigest()


def prepare_sample(entry: dict[str, Any], denoised: Array | None, output: Path) -> None:
    """Publish native raw samples and supplied linear ACEScg denoise, never synthesize IDs.

    Borrowed image is copied to atomic array files; output is a cache for this entry only.
    The denoised beauty must come from the renderer/denoiser, not a noisy fallback.
    """
    shot = Path(entry["manifest"]).parent
    frame = f"{entry['frame']:04d}.exr"
    noisy = read_rgb(shot / "beauty" / frame)
    if denoised is None:
        denoised = read_rgb(shot / "beauty" / frame, part_name="denoised")
    if noisy.shape != denoised.shape or not np.isfinite(denoised).all():
        raise ValueError("Denoised beauty must be finite and match native beauty dimensions")
    sidecar = shot / "samples" / entry["id"]
    if sidecar.exists():
        save_samples(output / "deep", read_samples(sidecar))
    save_array(output / "noisy.npy", noisy)
    save_array(output / "denoised.npy", denoised.astype(np.float32))


@dataclass
class PreparedBatch:
    """Own native crop tensors and indices; single-caller lifetime through backward."""

    images: torch.Tensor
    target: dict[str, torch.Tensor]
    clean: list[int]
    live: list[int]
    selections: list[tuple[int, int, int]]
    scales: torch.Tensor
    normalized: torch.Tensor | None
    cached_inputs: dict[str, torch.Tensor] | None = None
    cached_pyramid: list[torch.Tensor] | None = None
    jobs: list[TorchJob] = field(default_factory=list[TorchJob])
    device_math: bool = False
    crop: int = 0
    halo: int = 0


def _tensor(values: list[Array], device: str) -> torch.Tensor:
    return torch.as_tensor(np.stack(values), device=device).float()


def _combine(
    clean: torch.Tensor | None, live: torch.Tensor | None, batch: PreparedBatch
) -> torch.Tensor:
    exemplar = clean if clean is not None else live
    assert exemplar is not None
    result = torch.empty(
        (len(batch.selections), *exemplar.shape[1:]), device=exemplar.device, dtype=torch.float32
    )
    for indices, value in ((batch.clean, clean), (batch.live, live)):
        if value is not None:
            result.index_copy_(0, torch.tensor(indices, device=result.device), value.float())
    return result


@lru_cache(maxsize=2)
def _shot(root: Path) -> Shot:
    return Shot(
        load_samples(root / "deep") if (root / "deep/shape.npy").exists() else None,
        np.load(root / "noisy.npy", mmap_mode="r"),
        np.load(root / "denoised.npy", mmap_mode="r"),
    )


class OnlineLoader:
    """Own memory-map references; RNG and buffers belong to each sequential caller."""

    def __init__(
        self,
        frames: list[Frame],
        samples: Path,
        clean_fraction: float = 0.4,
        augmentation: dict[str, Any] | None = None,
        eligibility_warning_fraction: float = 0.25,
    ) -> None:
        require_supervision([frame.entry for frame in frames], purpose="training", root=samples)
        self.frames = frames
        self.clean_fraction = clean_fraction
        values = augmentation or {}
        if "clean" in values:
            raise ValueError("Use clean_fraction, not augmentation.clean")
        self.augmentation = AugmentConfig.from_dict({**values, "clean": 0.0})
        self.samples = samples
        if not 0 <= eligibility_warning_fraction <= 1:
            raise ValueError("eligibility_warning_fraction must be in [0, 1]")
        self.eligible = frozenset(
            i for i, frame in enumerate(frames) if frame.entry["recomposition_eligible"]
        )
        self.eligible_fraction = len(self.eligible) / len(frames)
        self.eligible_indices = tuple(sorted(self.eligible))
        logger = logging.getLogger(__name__)
        logger.info(
            "Recomposition eligible views: %d/%d (%.1f%%)",
            len(self.eligible),
            len(frames),
            100 * self.eligible_fraction,
        )
        if self.eligible_fraction < eligibility_warning_fraction:
            logger.warning(
                "Recomposition eligibility %.1f%% is below configured threshold %.1f%%",
                100 * self.eligible_fraction,
                100 * eligibility_warning_fraction,
            )
        self.roles: list[Roles] = []
        for frame in frames:
            root = samples / frame.entry["id"]
            denoised = np.load(root / "denoised.npy", mmap_mode="r")
            context = json.loads(
                (frame.feature_root / frame.entry["id"] / "context.json").read_text()
            )
            if context.get("image_sha256") != image_digest(denoised):
                raise ValueError(
                    "Clean feature cache must match denoised sample; extract --samples"
                )
            if frame.entry["recomposition_eligible"] and not (root / "deep/shape.npy").exists():
                raise ValueError(f"Published raw samples missing for {frame.entry['id']}")
            shot = _shot(root)
            if (
                frame.entry["recomposition_eligible"] and shot.deep.shape != frame.base.shape
            ) or denoised.shape != (3, *frame.base.shape):
                raise ValueError(
                    "Raw samples, denoised image and targets must share native dimensions"
                )
            self.roles.append(roles(json.loads(Path(frame.entry["manifest"]).read_text())))

    def shot(self, index: int) -> Shot:
        return _shot(self.samples / self.frames[index].entry["id"])

    def prepare(
        self,
        selections: list[tuple[int, int, int]],
        rng: np.random.Generator,
        crop: int,
        halo: int,
        device: str,
    ) -> PreparedBatch:
        """Return owned crops/targets; caller owns RNG and serializes loader access."""
        images: list[Array] = []
        targets: dict[str, list[Array]] = {
            k: [] for k in ("inverse_depth", "alpha", "rgb", "beauty", "mixed")
        }
        clean: list[int] = []
        live: list[int] = []
        scales: list[float] = []
        size = crop + 2 * halo
        for position, (i, top, left) in enumerate(selections):
            frame = self.frames[i]
            window = (slice(top, top + size), slice(left, left + size))
            layers = Layers(
                *(frame.arrays[k][..., *window] for k in ("inverse_depth", "alpha", "rgb"))
            )
            if rng.random() < self.clean_fraction:
                clean.append(position)
                image = self.shot(i).denoised[:, *window]
                scale = frame.scale
            else:
                live.append(position)
                if i in self.eligible:
                    donor_i = self.eligible_indices[int(rng.integers(len(self.eligible_indices)))]
                    y, x = self.frames[donor_i].crop(rng, size)
                    donor = Donor(self.shot(donor_i).crop(y, x, size, size), self.roles[donor_i])
                    sample = augment_sample(
                        self.shot(i).crop(top, left, size, size),
                        layers,
                        donor,
                        rng,
                        self.augmentation,
                    )
                    image, layers = sample.image, sample.layers
                else:
                    image = photometric(
                        self.shot(i).denoised[:, *window], rng, self.augmentation, {}
                    )
                # Native cropping changes normalized width, not focal length in pixels.
                scale = frame.scale * size / frame.base.shape[1]
            images.append(image)
            scales.append(scale)
            values = {
                **layers.arrays(),
                "beauty": composite(layers.alpha, layers.rgb),
                "mixed": (layers.alpha[0] > 0.03) & (layers.alpha[0] < 0.97),
            }
            for key in targets:
                targets[key].append(values[key][..., halo : halo + crop, halo : halo + crop])
        normalized: torch.Tensor | None = None
        if live:
            display = [display_rgb(images[i], "ACEScg") for i in live]
            rgb = _tensor(display, device) / 255
            mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
            normalized = (rgb - mean) / std
        return PreparedBatch(
            _tensor(images, device),
            {k: _tensor(v, device) for k, v in targets.items()},
            clean,
            live,
            selections,
            torch.tensor(scales, device=device)[:, None, None, None],
            normalized,
        )

    def predict(
        self,
        prepared: PreparedBatch,
        encoder: FrozenEncoder,
        teacher: DeepDepthPro,
        model: torch.nn.Module,
        crop: int,
        halo: int,
        decoder: bool,
    ) -> dict[str, torch.Tensor]:
        """Borrow weights, return live gradients; caller serializes model updates."""
        device = prepared.images.device.type
        clean_inputs: dict[str, torch.Tensor] = {}
        clean_pyramid: list[torch.Tensor] = []
        if prepared.clean and prepared.cached_inputs is not None:
            clean_inputs = prepared.cached_inputs
            clean_pyramid = prepared.cached_pyramid or []
        elif prepared.clean:
            clean_inputs, clean_pyramid = input_batch(
                self.frames,
                [prepared.selections[i] for i in prepared.clean],
                crop,
                halo,
                device,
                decoder,
            )
        live_pyramid: list[torch.Tensor] = []
        live_features: torch.Tensor | None = None
        live_base: torch.Tensor | None = None
        if prepared.live:
            assert prepared.normalized is not None
            live_pyramid = encoder(prepared.normalized)
            if not decoder:
                with torch.no_grad():
                    live_features = teacher.decode(live_pyramid)
                    with torch.autocast(device, enabled=False):
                        prior: torch.Tensor = teacher.deep_head.inverse_head(live_features.float())
                    live_base = prior.clamp_min(1e-5)
                    live_base = live_base * prepared.scales[prepared.live]
        if decoder:
            pyramid = [
                _combine(
                    clean_pyramid[i] if clean_pyramid else None,
                    live_pyramid[i] if live_pyramid else None,
                    prepared,
                )
                for i in range(4)
            ]
            return teacher.from_pyramid(pyramid, prepared.images, prepared.scales)
        features = _combine(clean_inputs.get("features"), live_features, prepared)
        base = _combine(clean_inputs.get("base"), live_base, prepared)
        return cast(dict[str, torch.Tensor], model(features, prepared.images, base))
