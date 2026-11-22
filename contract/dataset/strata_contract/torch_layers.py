"""Optional Torch realization of the shared K-layer rule on validated sorted CSR samples."""

import math
from importlib import import_module
from typing import Any

import torch

from strata_contract.layers import GAP_RATIO, LAYERS

api: Any = import_module("torch")


def segment_sum(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    return api.segment_reduce(values, "sum", lengths=counts)


def transmittance(alpha: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Borrow validated CSR alpha/counts; return owned float64 per-sample transmission."""
    logs = torch.log1p(-alpha.double().clamp(0, 1 - 1e-15))
    cumulative = torch.cat((logs.new_zeros(1), logs.cumsum(0)))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    origin = torch.repeat_interleave(cumulative[offsets[:-1]], counts, output_size=alpha.numel())
    return (cumulative[:-1] - origin).exp()


def reduce_sorted_samples(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    rgb: torch.Tensor,
    counts: torch.Tensor,
    shape: tuple[int, int],
    sky: torch.Tensor,
    layers: int = LAYERS,
) -> dict[str, torch.Tensor]:
    """Own native float32 layer tensors; borrow validated stable depth-sorted CSR inputs.

    All tensors share a device; depth/alpha/RGB obey layers.reduce_samples semantics.
    The caller validates producer shape/counts/positive finite depths before device transfer.
    No global state is mutated; concurrent calls use independent buffers.
    """
    alpha = alpha.double().clamp(0, 1)
    trans = transmittance(alpha, counts)
    weights = trans * alpha
    keep = torch.nonzero(weights > 1e-12).flatten()
    pixel = torch.repeat_interleave(
        torch.arange(counts.numel(), device=depth.device), counts, output_size=depth.numel()
    )[keep]
    z, a, rgb = depth[keep].double(), alpha[keep], rgb[keep].double()
    t, w = trans[keep], weights[keep]
    counts = torch.bincount(pixel, minlength=counts.numel())
    first = torch.cat((counts.new_zeros(1), counts.cumsum(0)))[:-1]
    index = torch.arange(z.numel(), device=z.device)
    previous = (index - 1).clamp_min(0)
    log = z.log()
    gaps = log - log[previous]
    gaps = torch.where(index == first[pixel], -torch.inf, gaps)
    splits = torch.zeros_like(gaps, dtype=torch.bool)
    for _ in range(layers - 1):
        largest = torch.full((counts.numel(),), -torch.inf, device=z.device, dtype=z.dtype)
        largest.scatter_reduce_(0, pixel, gaps, reduce="amax", include_self=True)
        candidate = (gaps == largest[pixel]) & (gaps > math.log(GAP_RATIO))
        chosen = torch.full_like(counts, z.numel())
        chosen.scatter_reduce_(
            0, pixel, torch.where(candidate, index, z.numel()), reduce="amin", include_self=True
        )
        selected = index == chosen[pixel]
        splits |= selected
        gaps = torch.where(selected, -torch.inf, gaps)
    cumulative = torch.cat((counts.new_zeros(1), splits.long().cumsum(0)))
    slot = cumulative[1:] - cumulative[first[pixel]]
    group = pixel * layers + slot
    lengths = torch.bincount(group, minlength=counts.numel() * layers)
    maximum_t = api.segment_reduce(t, "max", lengths=lengths, initial=0.0)
    local = t / maximum_t[group].clamp_min(1e-300)
    visible = segment_sum(w, lengths)
    inverse = segment_sum(w / z, lengths) / visible.clamp_min(1e-300)
    sky = segment_sum(local * a * (sky[keep]), lengths)
    valid = (visible > 1e-12) & (sky.float() <= 1e-12)
    inverse = torch.where(valid, inverse, 0)
    coverage = visible / maximum_t.clamp_min(1e-300)
    colour = segment_sum(local[:, None] * rgb, lengths)
    coverage = torch.where(visible > 1e-12, coverage, 0).clamp(0, 1)
    colour = torch.where((visible > 1e-12)[:, None], colour, 0)
    height, width = shape
    return {
        "inverse_depth": inverse.reshape(height, width, layers).permute(2, 0, 1).float(),
        "alpha": coverage.reshape(height, width, layers).permute(2, 0, 1).float(),
        "rgb": colour.reshape(height, width, layers, 3).permute(2, 3, 0, 1).float(),
    }
