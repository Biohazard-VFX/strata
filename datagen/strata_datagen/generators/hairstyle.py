"""Seeded human hairstyles grown on a head's scalp and face regions, morphing smoothly over a shot.

A style is a cut (buzz to long), a texture (straight to coily), an updo (ponytail, bun, braid,
pigtails), flyaways and frizz, an optional wet look, facial hair and a melanin hair colour. Down styles
grow guide strands under gravity against the skin and interpolate dense strands from them; updos
lay every strand along the scalp to a tie and on into a tail, plait or bun.

Within a shot the cut and updo stay fixed while length, curl, frizz, clumping, density and melanin drift
as smooth seeded functions of time. Strand count and samples per strand never change: strands are
shaped once at their longest, then trimmed and textured per key time, and density thins by fading
widths to zero, so time-sampled points interpolate and motion blur and deep stay valid. The scalp's
points over all keys are capped; capped styles grow fewer, wider strands with the same summed width.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom
from scipy.spatial import KDTree

from strata_datagen import strands, timeline
from strata_datagen.usdutil import FloatArray, IntArray, floats, ints, vec3f

VERSION = "hairstyle-0.2.0"
REGIONS = ("scalp", "brows", "lashes", "beard")
CUTS = ("buzz", "short", "medium", "long")
CUT_WEIGHTS = (0.12, 0.3, 0.28, 0.3)
TEXTURES = ("straight", "wavy", "curly", "coily")
TEXTURE_WEIGHTS = (0.4, 0.25, 0.2, 0.15)
UPDOS = ("down", "ponytail", "bun", "braid", "pigtails")
UPDO_WEIGHTS = (0.55, 0.2, 0.1, 0.1, 0.05)
FACIAL = ("none", "stubble", "short_beard", "full_beard")
FACIAL_WEIGHTS = (0.45, 0.25, 0.15, 0.15)
COLOURS: tuple[tuple[str, float, tuple[float, float], tuple[float, float]], ...] = (
    ("black", 0.3, (0.85, 0.98), (0.05, 0.15)),
    ("dark_brown", 0.25, (0.6, 0.85), (0.1, 0.3)),
    ("brown", 0.15, (0.4, 0.6), (0.15, 0.4)),
    ("light_brown", 0.1, (0.25, 0.4), (0.2, 0.45)),
    ("blonde", 0.08, (0.08, 0.25), (0.25, 0.5)),
    ("red", 0.05, (0.3, 0.6), (0.7, 1.0)),
    ("grey", 0.07, (0.0, 0.08), (0.0, 0.3)),
)
LENGTH_M = {"buzz": (0.003, 0.012), "short": (0.03, 0.09), "medium": (0.1, 0.25), "long": (0.28, 0.65)}
CURL = {
    "straight": (0.0, 0.0, 0.0, 0.0),
    "wavy": (0.004, 0.012, 3.0, 8.0),
    "curly": (0.004, 0.01, 15.0, 35.0),
    "coily": (0.0015, 0.004, 60.0, 150.0),
}
DRIFTED = ("length", "curl", "frizz", "clump", "density", "melanin", "facial_length")
DRIFT_MAX = (0.35, 0.5, 0.8, 0.4, 0.35, 0.15, 0.4)
KEY_SECONDS = 6.0
FADE = 0.05
SKIN_OFFSET_M = 0.002
MAX_POINTS = 6_000_000
SHOT_POINTS = 9_000_000
MAX_WIDTH_SCALE = 4.0


@dataclass(frozen=True)
class Head:
    """Skin triangles in pose space with per-vertex region weights; up and forward are pose-space axes."""

    points: FloatArray
    triangles: IntArray
    regions: dict[str, FloatArray]
    up: FloatArray
    forward: FloatArray


@dataclass(frozen=True)
class Colour:
    name: str
    melanin: float
    redness: float
    randomize: float


@dataclass(frozen=True)
class Drift:
    """Per DRIFTED parameter a timeline curve of offsets (relative; absolute for melanin)."""

    curves: tuple[timeline.Curve, ...]
    wave_m: float
    wave_turns_per_m: float

    def wave(self, name: str, seconds: float) -> float:
        return float(self.curves[DRIFTED.index(name)](np.array([seconds]))[0])

    def span(self, name: str) -> float:
        return float(np.abs(self.curves[DRIFTED.index(name)].keys).max())


STILL = Drift(tuple(timeline.Curve(np.zeros(4), 1.0) for _ in DRIFTED), 0.0, 0.0)


@dataclass(frozen=True)
class Style:
    cut: str
    texture: str
    updo: str
    part: str
    bangs: bool
    length_m: float
    density_per_cm2: float
    width_m: float
    volume: float
    curl_radius_m: float
    curl_turns_per_m: float
    clump: float
    clump_size: int
    frizz_m: float
    flyaways: float
    wet: bool
    tie_elevation_deg: float
    colour: Colour
    facial: str
    facial_length_m: float
    facial_colour: Colour
    brows: bool
    lashes: bool
    drift: Drift = STILL


@dataclass(frozen=True)
class StrandSet:
    """Curves sharing one sample count per key time: (keys, curves, samples, 3) points and widths."""

    name: str
    curves: FloatArray
    widths: FloatArray


@dataclass(frozen=True)
class Hairstyle:
    style: Style
    seconds: tuple[float, ...]
    sets: tuple[StrandSet, ...]


def _drift(rng: np.random.Generator, duration_s: float) -> Drift:
    curves = tuple(
        timeline.curve(rng, duration_s, float(rng.uniform(8.0, 20.0)), -a, a)
        for a in (float(rng.uniform(0.0, m)) for m in DRIFT_MAX)
    )
    wave = float(rng.uniform(0.002, 0.006)) if rng.random() < 0.4 else 0.0
    return Drift(curves, wave, float(rng.uniform(3.0, 8.0)))


def record(style: Style) -> dict[str, object]:
    """The style as JSON-ready values, drift curves as their keys and periods."""
    out = asdict(style)
    out["drift"] = {
        "curves": {
            n: {"keys": c.keys.tolist(), "period_s": c.period_s}
            for n, c in zip(DRIFTED, style.drift.curves, strict=True)
        },
        "wave_m": style.drift.wave_m,
        "wave_turns_per_m": style.drift.wave_turns_per_m,
    }
    return out


def at(style: Style, seconds: float) -> Style:
    """The style `seconds` into the shot; straight hair may blend into waves and back."""
    d = style.drift
    swell = max(0.0, d.wave("curl", seconds) / max(d.span("curl"), 1e-9))
    curled = style.curl_radius_m > 0.0
    colour = replace(
        style.colour, melanin=float(np.clip(style.colour.melanin + d.wave("melanin", seconds), 0.0, 1.0))
    )
    return replace(
        style,
        length_m=style.length_m * (1.0 + d.wave("length", seconds)),
        curl_radius_m=style.curl_radius_m * (1.0 + d.wave("curl", seconds)) if curled else d.wave_m * swell,
        curl_turns_per_m=style.curl_turns_per_m if curled else d.wave_turns_per_m,
        frizz_m=style.frizz_m * (1.0 + d.wave("frizz", seconds)),
        clump=float(np.clip(style.clump * (1.0 + d.wave("clump", seconds)), 0.0, 0.98)),
        density_per_cm2=style.density_per_cm2 * (1.0 + d.wave("density", seconds)),
        colour=colour,
        facial_length_m=style.facial_length_m * (1.0 + d.wave("facial_length", seconds)),
    )


def key_seconds(duration_s: float) -> tuple[float, ...]:
    """Key times every KEY_SECONDS plus the last instant; points between keys interpolate linearly."""
    keys = np.arange(0.0, duration_s, KEY_SECONDS).tolist()
    return (*keys, duration_s) if duration_s > 0.0 else (0.0,)


def _pick(rng: np.random.Generator, options: tuple[str, ...], weights: tuple[float, ...]) -> str:
    p = np.asarray(weights) / np.sum(weights)
    return options[int(rng.choice(len(options), p=p))]


def _colour(rng: np.random.Generator) -> Colour:
    weights = np.array([c[1] for c in COLOURS])
    name, _, melanin, redness = COLOURS[int(rng.choice(len(COLOURS), p=weights / weights.sum()))]
    return Colour(
        name, float(rng.uniform(*melanin)), float(rng.uniform(*redness)), float(rng.uniform(0.03, 0.15))
    )


def _facial_colour(rng: np.random.Generator, hair: Colour) -> Colour:
    if hair.name != "grey" and rng.random() < 0.15:
        return Colour("grey", float(rng.uniform(0.0, 0.1)), hair.redness, 0.3)
    melanin = float(np.clip(hair.melanin + rng.uniform(-0.05, 0.1), 0.0, 1.0))
    return Colour(hair.name, melanin, hair.redness, hair.randomize)


def sample(
    rng: np.random.Generator, regions: tuple[str, ...], *, facial_hair: bool, duration_s: float = 0.0
) -> Style:
    """Draw a style; every draw happens whatever the head offers, so a seed keeps its other choices."""
    cut = _pick(rng, CUTS, CUT_WEIGHTS)
    texture = _pick(rng, TEXTURES, TEXTURE_WEIGHTS)
    updo = _pick(rng, UPDOS, UPDO_WEIGHTS)
    updo = updo if cut in ("medium", "long") else "down"
    part = _pick(rng, ("none", "centre", "side"), (0.4, 0.3, 0.3))
    bangs = bool(rng.random() < 0.3) and cut in ("medium", "long") and updo == "down"
    length = float(rng.uniform(*LENGTH_M[cut]))
    radius_lo, radius_hi, turns_lo, turns_hi = CURL[texture]
    curl_radius = float(rng.uniform(radius_lo, radius_hi)) if radius_hi > 0.0 else 0.0
    curl_turns = float(rng.uniform(turns_lo, turns_hi)) if turns_hi > 0.0 else 0.0
    wet = bool(rng.random() < 0.12)
    clump_scale = 0.3 if texture == "coily" else 1.0
    volume = float(rng.uniform(0.6, 0.95)) if texture == "coily" else float(rng.uniform(0.1, 0.45))
    colour = _colour(rng)
    facial = _pick(rng, FACIAL, FACIAL_WEIGHTS)
    facial_length = {
        "none": 0.0,
        "stubble": float(rng.uniform(0.0008, 0.004)),
        "short_beard": float(rng.uniform(0.006, 0.025)),
        "full_beard": float(rng.uniform(0.03, 0.12)),
    }[facial]
    facial_colour = _facial_colour(rng, colour)
    return Style(
        cut=cut,
        texture=texture,
        updo=updo,
        part=part,
        bangs=bangs,
        length_m=length,
        density_per_cm2=float(rng.uniform(70.0, 130.0)),
        width_m=float(rng.uniform(55e-6, 100e-6)),
        volume=volume,
        curl_radius_m=curl_radius,
        curl_turns_per_m=curl_turns,
        clump=float(rng.uniform(0.85, 0.97)) if wet else float(rng.uniform(0.2, 0.7)) * clump_scale,
        clump_size=int(rng.integers(15, 50)) if wet else int(rng.integers(40, 160)),
        frizz_m=float(rng.uniform(0.0, 0.0003)) if wet else float(rng.uniform(0.0001, 0.0015)),
        flyaways=float(rng.uniform(0.0, 0.005)) if wet else float(rng.uniform(0.003, 0.03)),
        wet=wet,
        tie_elevation_deg=float(rng.uniform(-25.0, 45.0)),
        colour=colour,
        facial=facial if facial_hair and "beard" in regions else "none",
        facial_length_m=facial_length,
        facial_colour=facial_colour,
        brows="brows" in regions,
        lashes="lashes" in regions,
        drift=_drift(rng, duration_s),
    )


@dataclass(frozen=True)
class _Frame:
    centre: FloatArray
    radius: float
    up: FloatArray
    forward: FloatArray
    side: FloatArray


class _Skin:
    """Nearest-vertex collision: points closer to the skin than an offset are pushed out along normals."""

    def __init__(self, head: Head) -> None:
        tri = head.points[head.triangles]
        face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals = np.zeros_like(head.points)
        for corner in range(3):
            np.add.at(normals, head.triangles[:, corner], face_normal)
        self.points = head.points
        self.normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.tree = KDTree(head.points)

    def push(self, points: FloatArray, offset: FloatArray | float) -> FloatArray:
        flat = points.reshape(-1, 3)
        _, index = self.tree.query(flat)
        normal = self.normals[index]
        depth = np.sum((flat - self.points[index]) * normal, axis=1)
        gap = np.broadcast_to(np.asarray(offset, dtype=np.float64), points.shape[:-1]).reshape(-1) - depth
        return (flat + normal * np.maximum(gap, 0.0)[:, None]).reshape(points.shape)


def _frame(head: Head) -> _Frame:
    weight = head.regions["scalp"]
    scalp = head.points[weight > 0.5]
    a = np.concatenate([2.0 * scalp, np.ones((len(scalp), 1))], axis=1)
    solution = np.linalg.lstsq(a, np.sum(scalp**2, axis=1), rcond=None)[0]
    centre = solution[:3]
    radius = float(np.sqrt(solution[3] + centre @ centre))
    up = head.up / np.linalg.norm(head.up)
    forward = head.forward - up * (head.forward @ up)
    forward /= np.linalg.norm(forward)
    return _Frame(centre, radius, up, forward, np.cross(up, forward))


def _sample_region(
    head: Head, skin: _Skin, weight: FloatArray, count: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Roots spread by area and region weight (accepted by the interpolated weight), with skin normals."""
    tri = head.points[head.triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    corner = weight[head.triangles]
    p = area * corner.max(axis=1)
    found: list[FloatArray] = []
    normals: list[FloatArray] = []
    total = 0
    while total < count:
        face = rng.choice(len(p), 2 * count, p=p / p.sum())
        u, v = rng.random(2 * count), rng.random(2 * count)
        flip = u + v > 1.0
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        bary = np.stack([1.0 - u - v, u, v], axis=-1)
        keep = rng.random(2 * count) * corner[face].max(axis=1) < np.sum(bary * corner[face], axis=1)
        position = np.einsum("nk,nkd->nd", bary[keep], tri[face[keep]])
        normal: FloatArray = np.einsum("nk,nkd->nd", bary[keep], skin.normals[head.triangles[face[keep]]])
        found.append(position)
        normals.append(normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12))
        total += len(position)
    return np.concatenate(found)[:count], np.concatenate(normals)[:count]


def region_area_cm2(head: Head, weight: FloatArray) -> float:
    tri = head.points[head.triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(np.sum(area * weight[head.triangles].mean(axis=1)) * 1e4)


def _tangent(vectors: FloatArray, normals: FloatArray) -> FloatArray:
    out = vectors - normals * np.sum(vectors * normals, axis=-1, keepdims=True)
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)


def _local(frame: _Frame, points: FloatArray) -> FloatArray:
    d = (points - frame.centre) / frame.radius
    return np.stack([d @ frame.side, d @ frame.up, d @ frame.forward], axis=-1)


def _part_x(style: Style) -> float:
    return {"none": 0.0, "centre": 0.0, "side": 0.35}[style.part]


def _fringe(style: Style, q: FloatArray) -> NDArray[np.bool_]:
    """Roots along the front hairline that a fringe combs forward over the forehead."""
    if not style.bangs:
        return np.zeros(len(q), dtype=bool)
    return (q[:, 2] > 0.55) & (q[:, 1] > 0.0) & (np.abs(q[:, 0]) < 0.6)


def _comb(style: Style, frame: _Frame, roots: FloatArray, normals: FloatArray) -> FloatArray:
    """Root growth direction: away from the parting, back and down, lifted off the scalp by volume."""
    q = _local(frame, roots)
    if style.part == "none":
        lateral = np.zeros(len(q))
        back = np.full(len(q), 0.8)
    else:
        lateral = np.sign(q[:, 0] - _part_x(style) + 1e-9) * 0.8
        back = np.full(len(q), 0.25)
    if style.cut == "short":
        back = back - 1.0 * np.clip(q[:, 2], 0.0, 1.0)
    fringe = _fringe(style, q)
    back = np.where(fringe, -1.5, back)
    lateral = np.where(fringe, lateral * 0.3, lateral)
    want = frame.side[None] * lateral[:, None] - frame.forward[None] * back[:, None] - frame.up[None] * 0.4
    along = _tangent(want, normals)
    rise = 0.05 + (0.4 if style.cut == "short" else 0.9) * style.volume**2
    lift = np.where(fringe, 0.0, rise)[:, None]
    out = along * (1.0 - lift) + normals * lift
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _lengths(style: Style, frame: _Frame, roots: FloatArray, rng: np.random.Generator) -> FloatArray:
    q = _local(frame, roots)
    base = np.full(len(roots), style.length_m)
    if style.cut == "short":
        height = np.clip((q[:, 1] + 0.2) / 0.8, 0.0, 1.0)
        base = base * (0.3 + 0.7 * height * height * (3.0 - 2.0 * height))
    if style.cut in ("medium", "long"):
        base = base * (1.0 - 0.25 * np.clip(q[:, 2], 0.0, 1.0))
    fringe = _fringe(style, q)
    base = np.where(fringe, np.minimum(base, frame.radius * rng.uniform(0.9, 1.3)), base)
    return base * rng.uniform(0.85, 1.0, len(roots))


def _grow(
    skin: _Skin,
    frame: _Frame,
    roots: FloatArray,
    directions: FloatArray,
    lengths: FloatArray,
    *,
    samples: int,
    stiffness: float,
    offset: FloatArray,
) -> FloatArray:
    """Grow chains from roots, turning to gravity (stiffness is kept per cm) and pushed out of the skin."""
    step = lengths / (samples - 1)
    out = np.empty((len(roots), samples, 3))
    out[:, 0] = roots
    direction = directions.copy()
    keep = (stiffness ** (step / 0.01))[:, None]
    for k in range(1, samples):
        direction = direction * keep - frame.up[None] * (1.0 - keep)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        point = skin.push(out[:, k - 1] + direction * step[:, None], offset)
        moved = point - out[:, k - 1]
        direction = moved / np.maximum(np.linalg.norm(moved, axis=1, keepdims=True), 1e-12)
        out[:, k] = out[:, k - 1] + direction * step[:, None]
    return out


def _samples(style: Style, length: float, base: int) -> int:
    turns = style.curl_turns_per_m * length
    return int(np.clip(max(base, turns * 8.0 + 4.0), base, 160))


def _interpolate(
    roots: FloatArray, guide_roots: FloatArray, guides: FloatArray, side: IntArray, guide_side: IntArray
) -> FloatArray:
    """Each strand takes the inverse-distance blend of its nearest guides' shapes on its side of the part."""
    shapes = guides - guides[:, :1]
    out = np.empty((len(roots), guides.shape[1], 3))
    for label in np.unique(side):
        mine = side == label
        pool = np.flatnonzero(guide_side == label)
        if len(pool) == 0:
            pool = np.arange(len(guides))
        k = min(4, len(pool))
        distance, index = KDTree(guide_roots[pool]).query(roots[mine], k=k)
        distance = np.asarray(distance).reshape(int(mine.sum()), k)
        index = pool[np.asarray(index).reshape(int(mine.sum()), k)]
        weight = 1.0 / np.maximum(distance, 1e-5) ** 2
        weight /= weight.sum(axis=1, keepdims=True)
        out[mine] = roots[mine, None] + np.einsum("nk,nksd->nsd", weight, shapes[index])
    return out


def _clump_centres(roots: FloatArray, size: int, rng: np.random.Generator) -> IntArray:
    """Map every strand to a clump centre strand; centres map to themselves."""
    centres = rng.choice(len(roots), max(1, len(roots) // max(size, 1)), replace=False)
    _, index = KDTree(roots[centres]).query(roots)
    guide = centres[np.asarray(index, dtype=np.int64)]
    guide[centres] = centres
    return guide


def _texture(
    curves: FloatArray, roots: FloatArray, style: Style, rng: np.random.Generator, start: float
) -> FloatArray:
    reference = curves
    guide = _clump_centres(roots, style.clump_size, rng)
    if style.cut == "buzz":
        return strands.frizz(curves, distance=style.frizz_m * 0.3, shape=0.7, rng=rng)
    curves = strands.clump(
        curves, guide, factor=style.clump, shape=0.35, tip_spread=0.002 * (1.0 - style.clump), rng=rng
    )
    if style.curl_radius_m > 0.0:
        own = np.arange(len(curves))
        axis = own if style.texture in ("coily", "wavy") else guide
        curves = strands.curl(
            curves,
            axis,
            radius=style.curl_radius_m,
            turns_per_m=style.curl_turns_per_m,
            rng=rng,
            start=max(start, 0.05),
            factor_start=0.6,
            factor_end=1.0,
            random_phase=1.0 if axis is own else 0.15,
            handedness=float(rng.choice([-1.0, 1.0])),
        )
    tangle = style.texture == "coily"
    curves = strands.noise(
        curves,
        distance=0.01 if tangle else 0.002 + 0.006 * style.volume,
        scale=160.0 if tangle else 40.0,
        shape=0.4,
        rng=rng,
    )
    curves = strands.frizz(curves, distance=style.frizz_m, shape=0.7, rng=rng)
    return strands.restore_lengths(curves, reference) if style.texture == "straight" else curves


def _flyaways(curves: FloatArray, style: Style, rng: np.random.Generator) -> FloatArray:
    chosen = rng.random(len(curves)) < style.flyaways
    if not chosen.any():
        return curves
    out = curves.copy()
    loose = strands.frizz(curves[chosen], distance=0.0006, shape=0.8, rng=rng, cumulative=True)
    out[chosen] = strands.noise(loose, distance=0.008, scale=75.0, shape=0.85, rng=rng, offset_per_curve=0.05)
    return out


def _widths(curves: FloatArray, root_m: float, rng: np.random.Generator, taper: float = 0.25) -> FloatArray:
    t = strands.parameter(curves)
    scale = rng.uniform(0.85, 1.15, (len(curves), 1))
    profile = np.clip((1.0 - t) / taper, 0.0, 1.0)
    return root_m * scale * (0.15 + 0.85 * profile)


def _strand_count(style: Style, head: Head, samples: int, keys: int) -> tuple[int, float]:
    """Strands to grow and the width factor that keeps their summed width when the point cap binds.

    Scalp points are capped per key and per shot (SHOT_POINTS over all keys), so long morphing
    shots trade strand count, made up in width, for keys.
    """
    wanted = max(64, int(style.density_per_cm2 * region_area_cm2(head, head.regions["scalp"])))
    count = min(wanted, min(MAX_POINTS, SHOT_POINTS // keys) // samples)
    return count, min(wanted / count, MAX_WIDTH_SCALE)


@dataclass(frozen=True)
class _Base:
    """Strands shaped once at the longest the shot reaches, with what each key needs to finish them."""

    curves: FloatArray
    roots: FloatArray
    keep: FloatArray
    rank: FloatArray
    split: int
    taper: float
    seed: int
    width_scale: float


def _base(
    curves: FloatArray,
    roots: FloatArray,
    split: int,
    taper: float,
    rng: np.random.Generator,
    *,
    ragged: bool,
    width_scale: float,
) -> _Base:
    """Ragged ends vary strand lengths by up to 10%; tied strands keep equal lengths."""
    keep = rng.uniform(0.9, 1.0, len(curves)) if ragged else np.ones(len(curves))
    rank = rng.random(len(curves))
    return _Base(curves, roots, keep, rank, split, taper, int(rng.integers(2**31)), width_scale)


def _visible(rank: FloatArray, share: float) -> FloatArray:
    """Strands whose rank exceeds the visible share fade out through their width."""
    x = np.clip((share - rank) / FADE + 0.5, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _finish(
    skin: _Skin, base: _Base, style: Style, envelope: Style, *, down: bool
) -> tuple[FloatArray, FloatArray]:
    """One key of a strand set: shaped past `split` (an updo's tie), faded by density.

    Strands are trimmed to the key's length and textured with the shot's fixed random draws.
    """
    rng = np.random.default_rng(base.seed)
    share = style.length_m / envelope.length_m
    fixed, loose = base.curves[:, : base.split], base.curves[:, base.split :]
    loose = strands.trim(loose, base.keep * share)
    textured = style if down else replace(style, clump=style.clump * 0.5)
    loose = _texture(loose, loose[:, 0], textured, rng, start=0.08 if down else 0.02)
    curves = np.concatenate([fixed, loose], axis=1)
    curves = _flyaways(curves, style, rng)
    if down:
        curves = skin.push(curves, SKIN_OFFSET_M * 0.5)
    curves[:, 0] = base.roots
    widths = _widths(
        curves, style.width_m * base.width_scale, np.random.default_rng(base.seed + 1), taper=base.taper
    )
    return curves, widths * _visible(base.rank, style.density_per_cm2 / envelope.density_per_cm2)[:, None]


def _down(
    head: Head, skin: _Skin, frame: _Frame, style: Style, rng: np.random.Generator, *, keys: int
) -> _Base:
    weight = head.regions["scalp"]
    samples = _samples(style, style.length_m, 6 if style.cut == "buzz" else 32)
    count, width_scale = _strand_count(style, head, samples, keys)
    guide_roots, guide_normals = _sample_region(head, skin, weight, min(600, max(64, count // 60)), rng)
    lengths = _lengths(style, frame, guide_roots, rng)
    height = np.clip((_local(frame, guide_roots)[:, 1] + 0.3), 0.0, 1.3)
    offset = SKIN_OFFSET_M + height[:, None] * 0.012 * (0.3 + style.volume)
    stiffness = {"buzz": 0.99, "short": 0.93}.get(style.cut, 0.6 + 0.3 * style.volume)
    stiffness = 0.9 + 0.06 * style.volume if style.texture == "coily" else stiffness
    stiffness = stiffness * (0.8 if style.wet else 1.0)
    guides = _grow(
        skin,
        frame,
        guide_roots,
        _comb(style, frame, guide_roots, guide_normals),
        lengths,
        samples=samples,
        stiffness=stiffness,
        offset=offset[:, 0],
    )
    roots, _ = _sample_region(head, skin, weight, count, rng)
    part = _part_x(style)
    split = style.part != "none"
    side = (_local(frame, roots)[:, 0] > part).astype(np.int64) if split else np.zeros(count, int)
    guide_side = (
        (_local(frame, guide_roots)[:, 0] > part).astype(np.int64)
        if split
        else np.zeros(len(guide_roots), int)
    )
    curves = _interpolate(roots, guide_roots, guides, side, guide_side)
    return _base(curves, roots, 0, 0.25, rng, ragged=True, width_scale=width_scale)


def _ties(style: Style, frame: _Frame) -> FloatArray:
    if style.updo == "pigtails":
        return np.stack(
            [
                frame.centre + frame.radius * (s * 0.85 * frame.side - 0.35 * frame.forward - 0.2 * frame.up)
                for s in (-1.0, 1.0)
            ]
        )
    e = np.radians(style.tie_elevation_deg if style.updo != "bun" else abs(style.tie_elevation_deg) + 15.0)
    return (frame.centre + frame.radius * (-np.cos(e) * frame.forward + np.sin(e) * frame.up))[None]


def _scalp_paths(skin: _Skin, frame: _Frame, roots: FloatArray, ties: FloatArray, samples: int) -> FloatArray:
    """Slerp from each root to its tie around the head centre, kept just above the skin."""
    a, b = roots - frame.centre, ties - frame.centre
    ra, rb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ua, ub = a / ra[:, None], b / rb[:, None]
    omega = np.arccos(np.clip(np.sum(ua * ub, axis=1), -1.0, 1.0))[:, None]
    t = np.linspace(0.0, 1.0, samples)[None, :]
    sin = np.maximum(np.sin(omega), 1e-6)
    wa, wb = np.sin((1.0 - t) * omega) / sin, np.sin(t * omega) / sin
    direction = wa[..., None] * ua[:, None] + wb[..., None] * ub[:, None]
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    radius = ra[:, None] + (rb - ra)[:, None] * t
    path = frame.centre + direction * radius[..., None]
    lift = SKIN_OFFSET_M + 0.004 * np.sin(np.pi * t)
    path = skin.push(path, lift)
    path[:, 0] = roots
    return path


def _tail_axis(
    skin: _Skin, frame: _Frame, tie: FloatArray, style: Style, *, length: float, samples: int
) -> FloatArray:
    if style.updo == "bun":
        t = np.linspace(0.0, 1.0, samples)
        outward = (tie - frame.centre) / np.linalg.norm(tie - frame.centre)
        e1 = np.cross(outward, frame.side)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(outward, e1)
        turns = max(1.5, length / 0.12)
        radius = 0.03 * (1.0 - 0.6 * t)
        angle = 2.0 * np.pi * turns * t
        lift = 0.012 + 0.02 * np.sin(np.pi * t)
        ring = radius[:, None] * (np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2)
        return tie + ring + outward * lift[:, None]
    outward = (tie - frame.centre) / np.linalg.norm(tie - frame.centre)
    start = outward * 0.7 - frame.up * 0.3
    axis = _grow(
        skin,
        frame,
        tie[None],
        (start / np.linalg.norm(start))[None],
        np.array([length]),
        samples=samples,
        stiffness=0.5,
        offset=np.array([0.02]),
    )
    return axis[0]


def _tail(
    axis: FloatArray, count: int, style: Style, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Strands of one tie laid around the tail axis; returns the tail curves and their start points."""
    if style.updo == "braid" or (style.updo == "pigtails" and style.texture != "coily"):
        tail = strands.braid(
            np.zeros((count, len(axis), 3)),
            axis,
            radius=0.012,
            turns_per_m=float(rng.uniform(10.0, 16.0)),
            strand_radius=0.006,
            rng=rng,
        )
        return tail, tail[:, 0]
    batch = np.ascontiguousarray(axis[None])
    _, normal, binormal = strands.frames(batch)
    s = np.linspace(0.0, 1.0, len(axis))[None, :]
    radius = 0.008 + (0.025 if style.updo == "ponytail" else 0.012) * s**0.7
    angle = rng.uniform(0.0, 2.0 * np.pi, (count, 1))
    r = np.sqrt(rng.random((count, 1)))
    x, y = (r * np.cos(angle)) * radius, (r * np.sin(angle)) * radius
    tail = axis[None] + normal[0][None] * x[..., None] + binormal[0][None] * y[..., None]
    return tail, tail[:, 0]


def _updo(
    head: Head, skin: _Skin, frame: _Frame, style: Style, rng: np.random.Generator, *, keys: int
) -> _Base:
    weight = head.regions["scalp"]
    scalp_samples, tail_samples = 16, _samples(style, style.length_m, 32)
    count, width_scale = _strand_count(style, head, scalp_samples + tail_samples, keys)
    roots, _ = _sample_region(head, skin, weight, count, rng)
    ties = _ties(style, frame)
    which = np.argmin(np.linalg.norm(roots[:, None] - ties[None], axis=-1), axis=1)
    out = np.empty((count, scalp_samples + tail_samples - 1, 3))
    for i, tie in enumerate(ties):
        mine = np.flatnonzero(which == i)
        tail_length = max(0.06, style.length_m - 0.5 * np.pi * frame.radius * 0.6)
        axis = _tail_axis(skin, frame, tie, style, length=tail_length, samples=tail_samples)
        tail, start = _tail(axis, len(mine), style, rng)
        path = _scalp_paths(skin, frame, roots[mine], start, scalp_samples)
        out[mine] = np.concatenate([path, tail[:, 1:]], axis=1)
    return _base(out, roots, scalp_samples - 1, 0.15, rng, ragged=False, width_scale=width_scale)


def _facial(
    head: Head,
    skin: _Skin,
    frame: _Frame,
    name: str,
    *,
    count: int,
    length: float,
    samples: int,
    width: float,
    direction: FloatArray,
    rng: np.random.Generator,
) -> StrandSet:
    roots, normals = _sample_region(head, skin, head.regions[name], count, rng)
    q = _local(frame, roots)
    lateral: FloatArray = np.sign(q[:, 0])[:, None] * frame.side[None]
    want: FloatArray = (
        direction[0] * lateral + direction[1] * frame.up[None] + direction[2] * frame.forward[None]
    )
    along = _tangent(want, normals)
    grow: FloatArray = along * (1.0 - direction[3]) + normals * direction[3]
    grow /= np.linalg.norm(grow, axis=1, keepdims=True)
    lengths = length * rng.uniform(0.6, 1.0, count)
    curves = _grow(
        skin,
        frame,
        roots,
        grow,
        lengths,
        samples=samples,
        stiffness=0.8,
        offset=np.full(count, SKIN_OFFSET_M * 0.5),
    )
    return StrandSet(name, curves, _widths(curves, width, rng, taper=0.6))


def _beard(head: Head, skin: _Skin, frame: _Frame, style: Style, rng: np.random.Generator) -> StrandSet:
    """Beard strands at the longest the shot reaches; keys trim them back."""
    stubble = style.facial == "stubble"
    samples = 4 if stubble else _samples(style, style.facial_length_m, 12)
    beard = _facial(
        head,
        skin,
        frame,
        "beard",
        count=int(rng.uniform(40.0, 90.0) * region_area_cm2(head, head.regions["beard"])),
        length=style.facial_length_m,
        samples=samples,
        width=float(rng.uniform(90e-6, 140e-6)),
        direction=np.array([0.2, -1.0, 0.1, 0.5 if stubble else 0.3]),
        rng=rng,
    )
    if stubble:
        return beard
    curves = strands.frizz(beard.curves, distance=0.0015, shape=0.6, rng=rng)
    if style.curl_radius_m > 0.0:
        curves = strands.curl(
            curves,
            np.arange(len(curves)),
            radius=min(style.curl_radius_m, 0.004),
            turns_per_m=max(style.curl_turns_per_m, 40.0),
            rng=rng,
            start=0.1,
            random_phase=1.0,
        )
    return StrandSet("beard", skin.push(curves, SKIN_OFFSET_M * 0.5), beard.widths)


def _envelope(style: Style, keys: list[Style]) -> Style:
    """The longest, densest and most-curled state, which fixes counts and samples for the whole shot."""
    return replace(
        style,
        length_m=max(k.length_m for k in keys),
        density_per_cm2=max(k.density_per_cm2 for k in keys),
        curl_turns_per_m=max(k.curl_turns_per_m for k in keys),
        curl_radius_m=max(k.curl_radius_m for k in keys),
        facial_length_m=max(k.facial_length_m for k in keys),
    )


def _still(strand_set: StrandSet, keys: int) -> StrandSet:
    return StrandSet(
        strand_set.name,
        np.repeat(strand_set.curves[None], keys, axis=0),
        np.repeat(strand_set.widths[None], keys, axis=0),
    )


def build(
    head: Head, style: Style, rng: np.random.Generator, seconds: tuple[float, ...] = (0.0,)
) -> Hairstyle:
    """Strands at each key time (seconds into the shot), with one topology for all keys."""
    skin = _Skin(head)
    frame = _frame(head)
    keys = [at(style, t) for t in seconds]
    envelope = _envelope(style, keys)
    down = style.updo == "down"
    base = (_down if down else _updo)(head, skin, frame, envelope, rng, keys=len(seconds))
    shaped = [_finish(skin, base, key, envelope, down=down) for key in keys]
    sets = [StrandSet("scalp", np.stack([c for c, _ in shaped]), np.stack([w for _, w in shaped]))]
    if style.facial != "none":
        beard = _beard(head, skin, frame, envelope, rng)
        share = [k.facial_length_m / envelope.facial_length_m for k in keys]
        sets.append(
            StrandSet(
                "beard",
                np.stack([strands.trim(beard.curves, np.full(len(beard.curves), s)) for s in share]),
                np.repeat(beard.widths[None], len(keys), axis=0),
            )
        )
    facial: list[StrandSet] = []
    if style.brows:
        facial.append(
            _facial(
                head,
                skin,
                frame,
                "brows",
                count=int(rng.integers(400, 700)),
                length=float(rng.uniform(0.006, 0.012)),
                samples=6,
                width=float(rng.uniform(60e-6, 90e-6)),
                direction=np.array([1.0, 0.35, 0.0, 0.15]),
                rng=rng,
            )
        )
    if style.lashes:
        facial.append(
            _facial(
                head,
                skin,
                frame,
                "lashes",
                count=int(rng.integers(200, 320)),
                length=float(rng.uniform(0.007, 0.011)),
                samples=6,
                width=float(rng.uniform(80e-6, 120e-6)),
                direction=np.array([0.1, 0.6, 1.0, 0.2]),
                rng=rng,
            )
        )
    sets.extend(_still(f, len(keys)) for f in facial)
    return Hairstyle(style, seconds, tuple(sets))


def author(
    stage: Usd.Stage, path: str, hairstyle: Hairstyle, codes: tuple[float, ...] = ()
) -> dict[str, UsdGeom.BasisCurves]:
    """One linear BasisCurves prim per strand set under `path`.

    Points, widths and extents are time-sampled at `codes` (one per key), or defaults for a single key.
    """
    UsdGeom.Xform.Define(stage, path)
    times = [Usd.TimeCode(c) for c in codes] if codes else [Usd.TimeCode.Default()]
    out: dict[str, UsdGeom.BasisCurves] = {}
    for strand_set in hairstyle.sets:
        _, count, samples, _ = strand_set.curves.shape
        curves = UsdGeom.BasisCurves.Define(stage, f"{path}/{strand_set.name}")
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateCurveVertexCountsAttr(ints(np.full(count, samples)))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        points, widths, extent = (
            curves.CreatePointsAttr(),
            curves.CreateWidthsAttr(),
            curves.CreateExtentAttr(),
        )
        for time, key_points, key_widths in zip(times, strand_set.curves, strand_set.widths, strict=True):
            points.Set(vec3f(key_points.reshape(-1, 3)), time)
            widths.Set(floats(key_widths.ravel()), time)
            extent.Set(UsdGeom.Curves.ComputeExtent(points.Get(time), widths.Get(time)), time)
        curves.GetPrim().CreateAttribute("strata:hairstyle", Sdf.ValueTypeNames.String).Set(VERSION)
        out[strand_set.name] = curves
    return out
