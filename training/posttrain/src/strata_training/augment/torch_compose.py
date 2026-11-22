"""Device-side denoised deep recomposition and instance grading."""

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import torch

from strata_training.augment.deep import SPLIT_EPSILON
from strata_training.augment.pipeline import AugmentConfig, Roles
from strata_training.augment.torch_deep import TensorSamples, merge
from strata_training.torch_api import api
from strata_training.torch_layers import segment_sum


@dataclass(frozen=True)
class TensorShot:
    """Borrow immutable device tensors for one native crop; caller owns lifetime."""

    deep: TensorSamples | None
    noisy: torch.Tensor
    denoised: torch.Tensor


@dataclass(frozen=True)
class TensorLayer:
    samples: TensorSamples
    shot: TensorShot | None
    matrix: torch.Tensor


def _explained(shot: TensorShot, rgb: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    return (
        shot.denoised.double()
        * (rgb + SPLIT_EPSILON * coverage)
        / (shot.noisy.double().clamp_min(0) + SPLIT_EPSILON)
    )


def _flatten(samples: TensorSamples) -> tuple[torch.Tensor, torch.Tensor]:
    through = samples.through()
    h, w = samples.shape
    return (
        segment_sum(through[:, None] * samples.rgb.double(), samples.counts).T.reshape(3, h, w),
        segment_sum(through * samples.alpha.double(), samples.counts).reshape(h, w),
    )


def compose(
    layers: list[TensorLayer], environment: TensorShot, matrix: torch.Tensor
) -> tuple[torch.Tensor, TensorSamples]:
    """Return owned image and graded samples; preserve stable source depth ties."""
    merged, labels = merge([layer.samples for layer in layers])
    through = merged.through()
    h, w = merged.shape
    image = torch.zeros((3, h, w), device=merged.z.device, dtype=torch.float64)
    colours = merged.rgb.clone()
    inserted = torch.zeros_like(merged.alpha, dtype=torch.bool)
    for group, layer in enumerate(layers):
        mask = labels == group
        rgb = segment_sum(
            through[:, None] * merged.rgb.double() * mask[:, None], merged.counts
        ).T.reshape(3, h, w)
        coverage = segment_sum(through * merged.alpha.double() * mask, merged.counts).reshape(h, w)
        part = rgb if layer.shot is None else _explained(layer.shot, rgb, coverage)
        image += torch.einsum("ij,jhw->ihw", layer.matrix.double(), part)
        colours[mask] = merged.rgb[mask] @ layer.matrix.float().T
        if layer.shot is not environment:
            inserted |= mask
    logs = torch.log1p(-merged.alpha.double().clamp(0, 1 - 1e-15)) * inserted
    transmitted = segment_sum(logs, merged.counts).exp().reshape(h, w)
    assert environment.deep is not None
    own_rgb, own_coverage = _flatten(environment.deep)
    residual = _explained(
        environment, (environment.noisy - own_rgb) * transmitted, (1 - own_coverage) * transmitted
    )
    image += torch.einsum("ij,jhw->ihw", matrix.double(), residual)
    return image.float(), replace(merged, rgb=colours)


def geometry(samples: TensorSamples) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    near = api.segment_reduce(
        torch.where(~samples.sky, samples.z, torch.inf),
        "min",
        lengths=samples.counts,
    )
    far = api.segment_reduce(
        torch.where(~samples.sky, samples.z, -torch.inf),
        "max",
        lengths=samples.counts,
    )
    return near.double(), far.double(), near.isfinite()


def _grade(rng: np.random.Generator, config: AugmentConfig, device: torch.device) -> torch.Tensor:
    from strata_training.augment.pipeline import grade_matrix as cpu_grade

    return torch.tensor(cpu_grade(rng, config), device=device)


def recompose_samples(
    shot: TensorShot,
    shot_roles: Roles,
    targets: dict[str, torch.Tensor],
    donor: TensorShot | None,
    donor_roles: Roles | None,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | TensorSamples]:
    """Borrow tensors and caller RNG; return an owned native image and matching targets."""
    del shot_roles
    device = shot.denoised.device
    scene = torch.eye(3, device=device, dtype=torch.float64)
    if rng.random() < config.scene_grade:
        stops = rng.uniform(*config.exposure_stops)
        balance = rng.normal(0, config.white_balance_stops, 2)
        gains = 2.0 ** np.array([stops + balance[0], stops, stops + balance[1]])
        scene = torch.diag(torch.tensor(gains, device=device))
        record["scene_gain"] = gains.tolist()
    layers: list[TensorLayer] = []
    deep = shot.deep
    if rng.random() < config.instance_grade:
        assert deep is not None
        hero = deep.roles[:, 1] > 0
        present = cast(torch.Tensor, api.unique(deep.ids[(deep.ids >= 0) & ~hero])).cpu().numpy()
        if bool(hero.any()):
            present = np.concatenate((present, [-2]))
        if present.size:
            count = min(
                int(rng.integers(config.instance_count[0], config.instance_count[1] + 1)),
                present.size,
            )
            chosen = rng.choice(present, count, replace=False)
            remaining = torch.ones_like(deep.ids, dtype=torch.bool)
            for instance in chosen:
                mask = hero if instance == -2 else (deep.ids == int(instance)) & ~hero
                mask &= remaining
                remaining &= ~mask
                layers.append(TensorLayer(deep.select(mask), shot, _grade(rng, config, device)))
            layers.append(TensorLayer(deep.select(remaining), shot, torch.eye(3, device=device)))
            record["graded_instances"] = [int(i) for i in chosen]
    if donor is not None and rng.random() < config.recompose:
        assert deep is not None
        assert donor.deep is not None
        assert donor_roles is not None
        element = _element(deep, donor, donor_roles, rng, config, record)
        if element is not None:
            if not layers:
                layers.append(TensorLayer(deep, shot, torch.eye(3, device=device)))
            layers.append(element)
    if rng.random() < config.occluders:
        assert deep is not None
        if bool((~deep.sky).any()):
            from strata_training.augment.torch_occluders import make_occluders

            if not layers:
                layers.append(TensorLayer(deep, shot, torch.eye(3, device=device)))
            layers.append(
                TensorLayer(
                    make_occluders(shot, rng, config, record), None, torch.eye(3, device=device)
                )
            )
        else:
            record["occluders_skipped"] = "no_geometry"
    if not layers:
        rgb = torch.einsum("ij,kjhw->kihw", scene, targets["rgb"].double()).float()
        image = torch.einsum("ij,jhw->ihw", scene, shot.denoised.double()).float()
        return image, {**targets, "rgb": rgb}
    scoped = [replace(layer, matrix=scene @ layer.matrix.double()) for layer in layers]
    image, samples = compose(scoped, shot, scene)
    return image, samples


def recompose(
    shot: TensorShot,
    shot_roles: Roles,
    targets: dict[str, torch.Tensor],
    donor: TensorShot | None,
    donor_roles: Roles | None,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image, result = recompose_samples(
        shot, shot_roles, targets, donor, donor_roles, rng, config, record
    )
    return image, result.targets(targets["alpha"].shape[0]) if isinstance(
        result, TensorSamples
    ) else result


def _element(
    background: TensorSamples,
    donor: TensorShot,
    roles: Roles,
    rng: np.random.Generator,
    config: AugmentConfig,
    record: dict[str, Any],
) -> TensorLayer | None:
    assert donor.deep is not None
    deep = donor.deep
    device = deep.z.device
    hero = deep.roles[:, 1] > 0
    scatter = torch.zeros_like(hero)
    for first, count in roles.scatter:
        scatter |= (deep.ids >= first) & (deep.ids < first + count)
    scatter_ids = (
        cast(torch.Tensor, api.unique(api.unique_consecutive(deep.ids[scatter]))).cpu().numpy()
    )
    if bool(hero.any()) and (rng.random() < config.hero or not scatter_ids.size):
        chosen = hero
        record["element"] = "hero"
    elif scatter_ids.size:
        count = min(
            int(rng.integers(config.scatter_count[0], config.scatter_count[1] + 1)),
            scatter_ids.size,
        )
        picked = rng.choice(scatter_ids, count, replace=False)
        chosen = torch.isin(deep.ids, torch.tensor(picked, device=device))
        record["element"] = [int(i) for i in picked]
    else:
        return None
    deepest = api.segment_reduce(
        torch.where(chosen, deep.z, -torch.inf), "max", lengths=deep.counts
    )
    keep = chosen | (deep.z < deepest[deep.pixel])
    element = deep.select(keep)
    front = rng.random() < config.element_front
    near, far, valid = geometry(background)
    donor_near, donor_far, donor_valid = geometry(element)
    both = valid & donor_valid
    if not bool(both.any()):
        return None
    in_front = float((near[both] / donor_far[both]).min())
    if front:
        scale = in_front * float(rng.uniform(0.6, 0.95))
    else:
        ratio = far[both] / donor_near[both]
        spread = max(float(torch.quantile(ratio, 0.5)), in_front * 1.01)
        scale = float(np.exp(rng.uniform(np.log(in_front), np.log(spread))))
    record["element_depth_scale"] = scale
    matrix = torch.eye(3, device=device, dtype=torch.float64)
    if rng.random() < config.instance_grade:
        record["element_graded"] = True
        matrix = _grade(rng, config, device)
    return TensorLayer(replace(element, z=element.z * scale), donor, matrix)
