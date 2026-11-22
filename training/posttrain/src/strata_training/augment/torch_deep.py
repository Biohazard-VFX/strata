"""Device-owned CSR samples, stable merging and contract-equivalent target reduction."""

from dataclasses import dataclass
from functools import cached_property

import torch
from strata_dataset import roles

from strata_training.torch_layers import reduce_sorted_samples, transmittance


@dataclass(frozen=True)
class TensorSamples:
    """Own pixel-major, stable depth-sorted tensors; immutable and caller-serialized."""

    shape: tuple[int, int]
    counts: torch.Tensor
    z: torch.Tensor
    alpha: torch.Tensor
    rgb: torch.Tensor
    ids: torch.Tensor
    roles: torch.Tensor

    @cached_property
    def sky(self) -> torch.Tensor:
        return roles.sky(self.alpha, self.roles[:, 0])

    @cached_property
    def pixel(self) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(self.counts.numel(), device=self.z.device),
            self.counts,
            output_size=self.z.numel(),
        )

    def validate(self) -> None:
        """Reject nonfinite/range/order violations before GPU visibility can hide them."""
        pixel = self.pixel
        valid = torch.isfinite(self.z).all() & (self.z > 0).all()
        valid &= (
            torch.isfinite(self.alpha).all() & (self.alpha >= 0).all() & (self.alpha <= 1).all()
        )
        valid &= torch.isfinite(self.rgb).all()
        valid &= torch.isfinite(self.roles).all()
        valid &= (self.roles[:, 2] >= 0).all() & (self.roles[:, 2] <= self.roles[:, 1]).all()
        valid &= (self.roles[:, 1] <= self.roles[:, 0]).all() & (
            self.roles[:, 0] <= self.alpha
        ).all()
        valid &= ((self.z[1:] >= self.z[:-1]) | (pixel[1:] != pixel[:-1])).all()
        if not bool(valid):
            raise ValueError(
                "Raw samples require finite positive sorted depths, finite RGB and alpha in [0,1]"
            )

    def select(self, keep: torch.Tensor) -> "TensorSamples":
        indices = torch.nonzero(keep).flatten()
        pixel = self.pixel[indices]
        return TensorSamples(
            self.shape,
            torch.bincount(pixel, minlength=self.counts.numel()),
            self.z[indices],
            self.alpha[indices],
            self.rgb[indices],
            self.ids[indices],
            self.roles[indices],
        )

    def through(self) -> torch.Tensor:
        return transmittance(self.alpha, self.counts)

    def targets(self, layers: int = 4) -> dict[str, torch.Tensor]:
        return reduce_sorted_samples(
            self.z,
            self.alpha,
            self.rgb,
            self.counts,
            self.shape,
            self.sky,
            layers,
        )


def batch_samples(parts: list[TensorSamples]) -> TensorSamples:
    """Concatenate independent native grids vertically without mixing pixel segments."""
    if len(parts) == 1:
        return parts[0]
    return TensorSamples(
        (sum(p.shape[0] for p in parts), parts[0].shape[1]),
        *(
            torch.cat([getattr(p, key) for p in parts])
            for key in ("counts", "z", "alpha", "rgb", "ids", "roles")
        ),
    )


def merge(parts: list[TensorSamples]) -> tuple[TensorSamples, torch.Tensor]:
    """Borrow sorted parts on one grid; return owned stable depth-merged samples/labels."""
    if len(parts) == 1:
        return parts[0], torch.zeros_like(parts[0].ids)
    z = torch.cat([p.z for p in parts])
    pixels = torch.cat([p.pixel for p in parts])
    key = (pixels << 32) | z.float().view(torch.int32).long()
    order = torch.argsort(key, stable=True)
    labels = torch.cat([torch.full_like(p.ids, i) for i, p in enumerate(parts)])[order]
    return TensorSamples(
        parts[0].shape,
        torch.stack([p.counts for p in parts]).sum(0),
        z[order],
        torch.cat([p.alpha for p in parts])[order],
        torch.cat([p.rgb for p in parts])[order],
        torch.cat([p.ids for p in parts])[order],
        torch.cat([p.roles for p in parts])[order],
    ), labels
