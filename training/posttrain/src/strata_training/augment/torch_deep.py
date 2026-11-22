"""Device-owned CSR samples, stable merging and contract-equivalent target reduction."""

from dataclasses import dataclass

import torch
from strata_contract.torch_layers import reduce_sorted_samples, transmittance


@dataclass(frozen=True)
class TensorSamples:
    """Own pixel-major, stable depth-sorted tensors; immutable and caller-serialized."""

    shape: tuple[int, int]
    counts: torch.Tensor
    z: torch.Tensor
    alpha: torch.Tensor
    rgb: torch.Tensor
    ids: torch.Tensor

    @property
    def pixel(self) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(self.counts.numel(), device=self.z.device),
            self.counts,
            output_size=self.z.numel(),
        )

    def select(self, keep: torch.Tensor) -> "TensorSamples":
        pixel = self.pixel[keep]
        return TensorSamples(
            self.shape,
            torch.bincount(pixel, minlength=self.counts.numel()),
            self.z[keep],
            self.alpha[keep],
            self.rgb[keep],
            self.ids[keep],
        )

    def through(self) -> torch.Tensor:
        return transmittance(self.alpha, self.counts)

    def targets(self, layers: int = 4) -> dict[str, torch.Tensor]:
        return reduce_sorted_samples(
            self.z, self.alpha, self.rgb, self.counts, self.shape, self.ids == 0, layers
        )


def merge(parts: list[TensorSamples]) -> tuple[TensorSamples, torch.Tensor]:
    """Borrow sorted parts on one grid; return owned stable depth-merged samples/labels."""
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
    ), labels
