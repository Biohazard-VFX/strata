"""Operators on batches of hair strands, shape (curves, samples, 3), for the hairstyle and groom generators.

The deformers follow Blender's CC0 procedural hair node groups (Clump, Curl, Frizz, Noise, Smooth, Trim,
Braid); tests/test_strands.py checks them against reference outputs evaluated in Blender.
"""

import numpy as np

from strata_datagen.usdutil import FloatArray, IntArray

CURL_TURNS_PER_FREQUENCY = 3.0
FRIZZ_GAIN = 1.35
NOISE_GAIN = 0.85


def segment_lengths(curves: FloatArray) -> FloatArray:
    return np.linalg.norm(np.diff(curves, axis=1), axis=-1)


def arc_lengths(curves: FloatArray) -> FloatArray:
    """Cumulative length from the root at every sample, shape (curves, samples)."""
    steps = segment_lengths(curves)
    return np.concatenate([np.zeros((len(curves), 1)), np.cumsum(steps, axis=1)], axis=1)


def parameter(curves: FloatArray) -> FloatArray:
    """Arc-length fraction of every sample (Blender's spline parameter factor)."""
    lengths = arc_lengths(curves)
    return lengths / np.maximum(lengths[:, -1:], 1e-12)


def _interp_rows(x: FloatArray, xp: FloatArray, fp: FloatArray) -> FloatArray:
    """Row-wise linear interpolation: x (n, m) into increasing xp (n, k) with values fp (n, k, d)."""
    n, k = int(xp.shape[0]), int(xp.shape[1])
    low = min(float(xp.min()), float(x.min()))
    span = max(float(xp.max()), float(x.max())) - low + 1.0
    shift = np.arange(n, dtype=np.float64)[:, None] * span
    flat: IntArray = np.searchsorted((xp - low + shift).ravel(), (x - low + shift).ravel(), side="right")
    index: IntArray = np.clip(flat.reshape(x.shape) - np.arange(n)[:, None] * k, 1, k - 1)
    rows: IntArray = np.arange(n)[:, None]
    x0, x1 = xp[rows, index - 1], xp[rows, index]
    w = np.clip((x - x0) / np.maximum(x1 - x0, 1e-12), 0.0, 1.0)[..., None]
    return fp[rows, index - 1] * (1.0 - w) + fp[rows, index] * w


def resample(curves: FloatArray, samples: int) -> FloatArray:
    """Every curve resampled to `samples` points evenly spaced by arc length."""
    lengths = arc_lengths(curves)
    targets = np.linspace(0.0, 1.0, samples)[None, :] * lengths[:, -1:]
    return _interp_rows(targets, lengths, curves)


def sample_at(curves: FloatArray, index: IntArray, factor: FloatArray) -> FloatArray:
    """Points on curves[index] at arc-length fractions `factor` (n, m)."""
    chosen = curves[index]
    return _interp_rows(factor, parameter(chosen), chosen)


def tangents(curves: FloatArray) -> FloatArray:
    steps = np.diff(curves, axis=1)
    steps /= np.maximum(np.linalg.norm(steps, axis=-1, keepdims=True), 1e-12)
    out = np.concatenate([steps[:, :1], steps[:, :-1] + steps[:, 1:], steps[:, -1:]], axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


def _take(values: FloatArray, index: IntArray) -> FloatArray:
    return np.take(values, index, axis=0)


def _dot(a: FloatArray, b: FloatArray) -> FloatArray:
    return np.einsum("...d,...d->...", a, b)


def _unit(v: FloatArray) -> FloatArray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _orthogonal(v: FloatArray) -> FloatArray:
    """Blender's math::orthogonal: a unit vector perpendicular to v chosen by v's dominant axis."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    axis = np.argmax(np.abs(v), axis=1)
    out = np.where(
        (axis == 0)[:, None],
        np.stack([-y - z, x, x], -1),
        np.where((axis == 1)[:, None], np.stack([y, -x - z, y], -1), np.stack([z, z, -x - y], -1)),
    )
    return _unit(out)


def frames(curves: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tangent, normal and binormal per sample, normals parallel-transported from the root (minimum twist)."""
    tangent = tangents(curves)
    normal = np.empty_like(tangent)
    normal[:, 0] = _orthogonal(tangent[:, 0])
    for i in range(1, curves.shape[1]):
        previous = normal[:, i - 1]
        projected = previous - tangent[:, i] * np.sum(previous * tangent[:, i], axis=-1, keepdims=True)
        normal[:, i] = _unit(projected)
    return tangent, normal, np.cross(tangent, normal)


def shape_range(t: FloatArray, shape: float, base: float = 2.0) -> FloatArray:
    """Blender's .shape_range profile over t in [0, 1].

    0 is constant 1 and 0.5 is t; towards 1 the rise moves to the tip. Negative shapes mirror it so
    the influence falls from the root instead (-0.5 is 1 - t).
    """
    if shape == 0.0:
        return np.ones_like(t)
    strength = min(abs(shape), 0.999999)
    folded = strength if strength <= 0.5 else 1.0 - strength
    b = max(base, 1.0000101)
    exponent = -np.log(max(1.0 - 2.0 * (1.0 - 1.0 / b) * folded, 1e-5)) / np.log(b)
    late = strength > 0.5
    x = np.clip(1.0 - t if (shape < 0.0) != late else t, 0.0, 1.0)
    y = np.power(x, exponent)
    return 1.0 - y if late else y


def _smootherstep(x: FloatArray) -> FloatArray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _random_unit_disc(rng: np.random.Generator, count: int) -> FloatArray:
    return rng.uniform(-1.0, 1.0, (count, 2))


def clump(
    curves: FloatArray,
    guide: IntArray,
    *,
    factor: float,
    shape: float,
    tip_spread: float,
    rng: np.random.Generator,
    clump_offset: float = 0.0,
    falloff: float = 0.0,
    threshold: float = 0.0,
) -> FloatArray:
    """Pull every curve towards its guide curve, keeping its root offset carried along the guide's frame.

    `guide` holds, per curve, the index of its guide within `curves`; guides map to themselves.
    """
    t = parameter(curves)
    target = sample_at(curves, guide, t)
    _, normal, binormal = frames(curves)
    root_offset = curves[:, 0] - _take(curves, guide)[:, 0]
    local = np.stack(
        [_dot(root_offset, _take(normal, guide)[:, 0]), _dot(root_offset, _take(binormal, guide)[:, 0])], -1
    )
    guide_normal = _interp_rows(t, parameter(curves[guide]), normal[guide])
    guide_binormal = _interp_rows(t, parameter(curves[guide]), binormal[guide])
    carried = guide_normal * local[:, None, :1] + guide_binormal * local[:, None, 1:]
    spread = _random_unit_disc(rng, len(curves)) * tip_spread
    shift = _random_unit_disc(rng, len(curves))[guide] * clump_offset
    tip = guide_normal * (spread + shift)[:, None, :1] + guide_binormal * (spread + shift)[:, None, 1:]
    weight = shape_range(t, shape)[..., None]
    goal = target + carried * (1.0 - weight) + tip * weight
    distance = np.sqrt(_dot(root_offset, root_offset))
    influence: FloatArray = np.ones(len(curves))
    if falloff > 0.0:
        ramp: FloatArray = np.clip((distance - threshold) / falloff, 0.0, 1.0)
        influence = 1.0 - ramp * ramp * (3.0 - 2.0 * ramp)
    mix: FloatArray = factor * weight * influence[:, None, None]
    return curves + (goal - curves) * mix


def curl(
    curves: FloatArray,
    guide: IntArray,
    *,
    radius: float,
    turns_per_m: float,
    rng: np.random.Generator,
    start: float = 0.0,
    factor_start: float = 1.0,
    factor_end: float = 1.0,
    random_phase: float = 0.0,
    handedness: float = 1.0,
) -> FloatArray:
    """Wind every curve on a helix of `radius` around its guide, starting `start` along the curve."""
    t = parameter(curves)
    lengths = _interp_rows(t, parameter(curves[guide]), arc_lengths(curves[guide])[..., None])[..., 0]
    _, normal, binormal = frames(curves)
    guide_normal = _interp_rows(t, parameter(curves[guide]), normal[guide])
    guide_binormal = _interp_rows(t, parameter(curves[guide]), binormal[guide])
    phase = rng.uniform(0.0, 1.0, len(curves)) * random_phase
    angle = 2.0 * np.pi * (handedness * turns_per_m * lengths + phase[:, None])
    scale = radius * (factor_start + (factor_end - factor_start) * t)
    offset = (guide_normal * np.sin(angle)[..., None] + guide_binormal * np.cos(angle)[..., None]) * scale[
        ..., None
    ]
    weight = np.sqrt(_smootherstep(t / start)) if start > 0.0 else np.ones_like(t)
    target = sample_at(curves, guide, t) + offset
    return curves + (target - curves) * weight[..., None]


def frizz(
    curves: FloatArray, *, distance: float, shape: float, rng: np.random.Generator, cumulative: bool = False
) -> FloatArray:
    """Random per-point displacement in each curve's frame, weighted along the curve by `shape`.

    Cumulative offsets add up from the root like Blender's default, so strands wander rather than jitter.
    """
    tangent, normal, binormal = frames(curves)
    r = rng.uniform(-1.0, 1.0, (*curves.shape[:2], 3)) * FRIZZ_GAIN
    offset = tangent * r[..., :1] + binormal * r[..., 1:2] + normal * r[..., 2:]
    offset = offset * distance * shape_range(parameter(curves), shape)[..., None]
    if cumulative:
        offset = np.cumsum(offset, axis=1)
    out = curves + offset
    out[:, 0] = curves[:, 0]
    return out


def _wave_field(points: FloatArray, scale: float, rng: np.random.Generator, waves: int = 12) -> FloatArray:
    """A smooth, zero-mean vector field from random sinusoids; deterministic per generator."""
    directions = _unit(rng.normal(size=(waves, 3)))
    frequency = scale * rng.uniform(0.5, 1.5, waves)
    phase = rng.uniform(0.0, 2.0 * np.pi, waves)
    amplitude = _unit(rng.normal(size=(waves, 3))) * np.sqrt(2.0 / waves)
    argument = np.einsum("...d,wd->...w", points, directions) * frequency + phase
    return np.einsum("...w,wd->...d", np.sin(argument), amplitude)


def noise(
    curves: FloatArray,
    *,
    distance: float,
    scale: float,
    shape: float,
    rng: np.random.Generator,
    offset_per_curve: float = 0.0,
    cumulative: bool = False,
) -> FloatArray:
    """Displace along a smooth spatial noise field, so neighbouring strands move together."""
    shift = rng.normal(size=(len(curves), 1, 3)) * offset_per_curve
    field = _wave_field(curves + shift, scale, rng) * NOISE_GAIN
    offset = field * distance * shape_range(parameter(curves), shape)[..., None]
    if cumulative:
        offset = np.cumsum(offset, axis=1)
    return curves + offset


def smooth(curves: FloatArray, *, amount: float, iterations: int) -> FloatArray:
    """Laplacian smoothing of interior points with root and tip pinned."""
    out = curves.copy()
    for _ in range(iterations):
        middle = 0.5 * (out[:, :-2] + out[:, 2:])
        out[:, 1:-1] += (middle - out[:, 1:-1]) * amount
    return out


def trim(curves: FloatArray, keep: FloatArray) -> FloatArray:
    """Cut every curve to the fraction `keep` (curves,) of its length, keeping the sample count."""
    lengths = arc_lengths(curves)
    targets = np.linspace(0.0, 1.0, curves.shape[1])[None, :] * (lengths[:, -1:] * keep[:, None])
    return _interp_rows(targets, lengths, curves)


def restore_lengths(curves: FloatArray, reference: FloatArray) -> FloatArray:
    """Rebuild each curve from its root with the reference's segment lengths along the new directions."""
    steps = np.diff(curves, axis=1)
    directions = _unit(steps)
    rebuilt = directions * segment_lengths(reference)[..., None]
    return np.concatenate([curves[:, :1], curves[:, :1] + np.cumsum(rebuilt, axis=1)], axis=1)


def braid(
    curves: FloatArray,
    axis: FloatArray,
    *,
    radius: float,
    turns_per_m: float,
    strand_radius: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Lay curves into a three-strand braid along one axis curve (samples, 3).

    Curve k goes to strand k % 3, which follows the figure-eight path of a plait; every hair keeps a
    random offset inside its strand.
    """
    count, samples, _ = curves.shape
    axis_batch = np.broadcast_to(axis, (1, samples, 3))
    _, normal, binormal = frames(np.ascontiguousarray(axis_batch))
    s = arc_lengths(np.ascontiguousarray(axis_batch))[0]
    strand = np.arange(int(count), dtype=np.int64) % 3
    phi: FloatArray = 2.0 * np.pi * turns_per_m * s[None, :] + 2.0 * np.pi * strand[:, None] / 3.0
    x = radius * np.sin(phi)
    y = 0.5 * radius * np.sin(2.0 * phi)
    disc = rng.normal(size=(count, 2))
    disc = disc / np.maximum(np.linalg.norm(disc, axis=1, keepdims=True), 1.0) * strand_radius
    x = x + disc[:, :1]
    y = y + disc[:, 1:]
    return axis[None] + normal[0][None] * x[..., None] + binormal[0][None] * y[..., None]


def nearest(points: FloatArray, targets: FloatArray) -> IntArray:
    """Index of the nearest target for every point (brute force in blocks; targets are guides)."""
    out = np.empty(len(points), dtype=np.int64)
    block = 4096
    for start in range(0, len(points), block):
        chunk = points[start : start + block]
        d = np.sum((chunk[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
        out[start : start + block] = np.argmin(d, axis=1)
    return out
