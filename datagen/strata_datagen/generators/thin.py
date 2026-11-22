"""Thin manufactured structures framed in a shot: wires, fences, nets, rigging, railings, antennae, grilles.

Each structure is placed in a depth band (in front of the camera, between camera and subject, or behind
the subject) at an anchor time along the camera path, so long shots pass many structures that enter
and leave the frame; its main element is drawn at a sub-pixel to few-pixel width. Flexible elements
(cables, fence wire, twine, rope) are linear BasisCurves; rigid ones (posts, rails, bars, masts) are
tube meshes. Hanging spans sway about their chord in a seeded wind and change sag with tension, as
smooth functions of time authored as time-sampled transforms; geometry never appears or vanishes.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial import KDTree

from strata_datagen import timeline
from strata_datagen.fields import Ground
from strata_datagen.usdutil import FloatArray, IntArray, floats, ints, matrix, tag_instances, vec3f

VERSION = "thin-0.1.0"
BANDS = ("foreground", "between", "background")
COUNT_WEIGHTS = (0.2, 0.4, 0.25, 0.15)
WINDOW_S = 8.0
GUST_PERIOD_S = 1.5
WIND_PERIOD_S = 10.0
TENSION_PERIOD_S = 8.0
PIXEL_WIDTH_RANGE = (0.15, 5.0)
MAX_MEMBER_PX = 24.0
MIN_MEMBER_RADIUS_M = 0.004
POINT_BUDGET = 200_000
ATTEMPTS = 16
TWIST_MIN_PX = 1.5
CHECK_STEP_M = 0.1
UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class Choice:
    """A structure's fixed kind and depth band; None means drawn from the seed."""

    kind: str | None = None
    band: str | None = None


@dataclass(frozen=True)
class View:
    """The shot a structure is framed in.

    cameras: camera_to_world per frame sample (column vectors, looking down -Z), as timeline samples;
    tan_half: tangents of the horizontal and vertical half angles; pixel_rad: the angle one pixel
    subtends; subject: the subject's ground positions, a cylinder it must stay clear of.
    """

    cameras: FloatArray
    tan_half: tuple[float, float]
    pixel_rad: float
    subject: FloatArray
    subject_radius_m: float
    subject_height_m: float

    @property
    def eyes(self) -> FloatArray:
        return self.cameras[:, :3, 3]


@dataclass(frozen=True)
class Swing:
    """Strands hanging between two anchors: they swing about the chord and their sag follows tension.

    The swing angle is swing_rad times a slow wind envelope in [0.3, 1] times a gust curve in [-1, 1];
    the sag (offset from the chord in the vertical plane) is scaled by 1 + sag_change * tension(t).
    """

    strands: tuple[int, ...]
    start: FloatArray
    end: FloatArray
    swing_rad: float
    sag_change: float
    gust: timeline.Curve
    wind: timeline.Curve
    tension: timeline.Curve

    def matrices(self, seconds: FloatArray) -> FloatArray:
        """Column-vector transforms of the rest strands, one per time."""
        axis = _unit(self.end - self.start)
        vertical = _unit(UP - (UP @ axis) * axis)
        across: FloatArray = np.cross(axis, vertical)
        angle = self.swing_rad * self.wind(seconds) * self.gust(seconds)
        sag = 1.0 + self.sag_change * self.tension(seconds)
        basis = np.stack([axis, vertical, across], axis=1)
        out = np.repeat(np.eye(4)[None], len(seconds), axis=0)
        for i, (theta, scale) in enumerate(zip(angle, sag, strict=True)):
            c, s = float(np.cos(theta)), float(np.sin(theta))
            local = np.array([[1.0, 0.0, 0.0], [0.0, c * scale, -s], [0.0, s * scale, c]])
            linear = basis @ local @ basis.T
            out[i, :3, :3] = linear
            out[i, :3, 3] = self.start - linear @ self.start
        return out


@dataclass(frozen=True)
class Finish:
    name: str
    color: tuple[float, float, float]
    metallic: float
    roughness: float


FINISHES = {
    f.name: f
    for f in (
        Finish("galvanised", (0.56, 0.57, 0.58), 1.0, 0.45),
        Finish("dark_steel", (0.30, 0.30, 0.31), 1.0, 0.35),
        Finish("copper", (0.80, 0.46, 0.32), 1.0, 0.35),
        Finish("rust", (0.30, 0.13, 0.06), 0.0, 0.85),
        Finish("black_coat", (0.025, 0.025, 0.025), 0.0, 0.4),
        Finish("green_coat", (0.03, 0.14, 0.05), 0.0, 0.45),
        Finish("white_paint", (0.75, 0.75, 0.73), 0.0, 0.5),
        Finish("red_paint", (0.45, 0.04, 0.03), 0.0, 0.5),
        Finish("yellow_paint", (0.70, 0.50, 0.03), 0.0, 0.5),
        Finish("blue_paint", (0.04, 0.12, 0.40), 0.0, 0.5),
        Finish("natural_fibre", (0.52, 0.42, 0.28), 0.0, 0.9),
        Finish("orange_fibre", (0.70, 0.25, 0.04), 0.0, 0.7),
        Finish("weathered_wood", (0.28, 0.20, 0.13), 0.0, 0.85),
    )
}
METALS = ("galvanised", "dark_steel", "rust")
COATED = ("galvanised", "black_coat", "green_coat")
PAINTED = ("galvanised", "dark_steel", "rust", "black_coat", "white_paint", "red_paint", "yellow_paint")


@dataclass(frozen=True)
class Member:
    """A rigid tube along a polyline; four sides make a square bar."""

    line: FloatArray
    radius_m: float
    sides: int


@dataclass(frozen=True)
class _Hang:
    strands: tuple[int, ...]
    start: FloatArray
    end: FloatArray
    swing_rad: float
    sag_change: float


@dataclass
class _Parts:
    strands: list[FloatArray] = field(default_factory=list[FloatArray])
    widths: list[float] = field(default_factory=list[float])
    members: list[Member] = field(default_factory=list[Member])
    hangs: list[_Hang] = field(default_factory=list[_Hang])

    def strand(self, line: FloatArray, width: float) -> int:
        self.strands.append(line)
        self.widths.append(width)
        return len(self.strands) - 1

    def hang(
        self, strands: tuple[int, ...], ends: tuple[FloatArray, FloatArray], swing_deg: float, sag: float
    ) -> None:
        self.hangs.append(_Hang(strands, ends[0], ends[1], float(np.radians(swing_deg)), sag))

    def member(self, line: FloatArray, radius: float, sides: int = 10) -> None:
        self.members.append(Member(line, radius, sides))


@dataclass(frozen=True)
class _Frame:
    """A structure's local frame on the ground: a along it, b across it (towards the camera), h up."""

    ground: Ground
    origin: FloatArray
    u: FloatArray
    n: FloatArray

    def at(self, a: object, h: object, b: object = 0.0) -> FloatArray:
        aa, hh, bb = np.broadcast_arrays(
            np.asarray(a, np.float64), np.asarray(h, np.float64), np.asarray(b, np.float64)
        )
        x = self.origin[0] + aa * self.u[0] + bb * self.n[0]
        z = self.origin[2] + aa * self.u[2] + bb * self.n[2]
        return np.stack([x, self.ground.height(x, z) + hh, z], axis=-1)

    def along(self, length: float, h: float, b: float = 0.0, step: float = 0.5) -> FloatArray:
        a = np.linspace(-0.5 * length, 0.5 * length, int(np.ceil(length / step)) + 1)
        return self.at(a, h, b)

    def post(self, a: float, top: float, b: float = 0.0) -> FloatArray:
        return self.at([a, a], [-0.3, top], [b, b])


@dataclass(frozen=True)
class _Plan:
    frame: _Frame
    length: float
    width: float
    width_px: float
    window: tuple[float, float]


@dataclass(frozen=True)
class Structure:
    kind: str
    band: str
    distance_m: float
    length_m: float
    width_m: float
    width_px: float
    strands: tuple[FloatArray, ...]
    strand_widths: tuple[float, ...]
    members: tuple[Member, ...]
    strand_finish: Finish
    member_finish: Finish
    anchor: int
    swings: tuple[Swing, ...]

    def points(self) -> FloatArray:
        """Every element sampled at most a few pixels apart, for clearance and framing checks."""
        step = min(CHECK_STEP_M, 0.02 * self.distance_m)
        lines = [*self.strands, *(m.line for m in self.members)]
        return np.concatenate([densify(line, step) for line in lines])

    def record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "band": self.band,
            "distance_m": round(self.distance_m, 4),
            "length_m": round(self.length_m, 4),
            "width_m": round(self.width_m, 6),
            "width_px_at_reference": round(self.width_px, 4),
            "strands": len(self.strands),
            "strand_points": sum(len(s) for s in self.strands),
            "members": len(self.members),
            "strand_finish": self.strand_finish.name,
            "member_finish": self.member_finish.name,
            "anchor_sample": self.anchor,
            "swinging_spans": len(self.swings),
            "max_swing_deg": round(
                float(np.degrees(max((w.swing_rad for w in self.swings), default=0.0))), 3
            ),
            "max_sag_change": round(max((w.sag_change for w in self.swings), default=0.0), 4),
        }


def _length(vector: FloatArray) -> float:
    return float(np.sqrt(np.sum(vector * vector)))


def _unit(vector: FloatArray) -> FloatArray:
    return vector / _length(vector)


def densify(line: FloatArray, step: float) -> FloatArray:
    segments = np.diff(line, axis=0)
    counts: IntArray = np.maximum(np.ceil(np.linalg.norm(segments, axis=1) / step).astype(np.int64), 1)
    index = np.repeat(np.arange(len(segments)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    fraction: FloatArray = (np.arange(int(counts.sum())) - first) / np.repeat(counts, counts)
    return np.vstack([line[index] + segments[index] * fraction[:, None], line[-1:]])


def catenary(start: FloatArray, end: FloatArray, sag_m: float, samples: int) -> FloatArray:
    """A hanging cable between two points; sag_m is its dip below the chord for level ends."""
    flat = end[[0, 2]] - start[[0, 2]]
    span = _length(flat)
    rise = float(end[1] - start[1])
    low, high = np.log(span / 200.0), np.log(span * span / (4.0 * sag_m) + span)
    for _ in range(100):
        a = np.exp(0.5 * (low + high))
        if a * (np.cosh(span / (2.0 * a)) - 1.0) > sag_m:
            low = 0.5 * (low + high)
        else:
            high = 0.5 * (low + high)
    a = float(np.exp(0.5 * (low + high)))
    x0 = 0.5 * span - a * np.arcsinh(rise / (2.0 * a * np.sinh(span / (2.0 * a))))
    x = np.linspace(0.0, span, samples)
    y = start[1] + a * (np.cosh((x - x0) / a) - np.cosh(x0 / a))
    t = x / span
    out = start[None] + t[:, None] * (end - start)[None]
    out[:, 1] = y
    out[-1] = end
    return out


def _frame_axes(tangent: FloatArray) -> tuple[FloatArray, FloatArray]:
    reference = UP if abs(float(tangent @ UP)) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(tangent, reference)
    normal /= np.linalg.norm(normal)
    return normal, np.cross(tangent, normal)


def twisted(centre: FloatArray, diameter: float) -> list[FloatArray]:
    """Three helical strands of a laid rope around a centre line."""
    pitch = 3.5 * diameter
    line = densify(centre, pitch / 8.0)
    steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
    arc = np.concatenate([np.zeros(1), np.cumsum(steps)])
    normal, binormal = _frame_axes(_unit(line[-1] - line[0]))
    strands: list[FloatArray] = []
    for k in range(3):
        angle = 2.0 * np.pi * (arc / pitch + k / 3.0)
        offset = np.cos(angle)[:, None] * normal + np.sin(angle)[:, None] * binormal
        strands.append(line + 0.3 * diameter * offset)
    return strands


def _rope(parts: _Parts, centre: FloatArray, plan: _Plan) -> tuple[int, ...]:
    if plan.width_px < TWIST_MIN_PX:
        return (parts.strand(centre, plan.width),)
    return tuple(parts.strand(strand, 0.55 * plan.width) for strand in twisted(centre, plan.width))


def _wires(rng: np.random.Generator, plan: _Plan) -> _Parts:
    span = rng.uniform(15.0, 45.0)
    sag = span * rng.uniform(0.008, 0.04)
    top = float(np.clip(rng.uniform(*plan.window), 0.6, 18.0)) + sag
    count = int(rng.integers(1, 7))
    stacked = rng.random() < 0.5
    offsets = np.arange(count) * rng.uniform(0.25, 0.8)
    heights = top + (offsets if stacked else 0.0 * offsets)
    across = (offsets - offsets.mean()) * (0.0 if stacked else 1.0)
    first = -0.5 * plan.length - rng.uniform(0.0, span)
    poles = first + span * np.arange(int(np.ceil((plan.length - first - 0.5 * plan.length) / span)) + 2)
    parts = _Parts()
    swing, tension = rng.uniform(1.0, 8.0), rng.uniform(0.03, 0.2)
    for a0, a1 in itertools.pairwise(poles):
        for h, b in zip(heights, across, strict=True):
            ends = plan.frame.at([a0, a1], [h, h], [b, b])
            cable = parts.strand(catenary(ends[0], ends[1], sag * rng.uniform(0.9, 1.1), 64), plan.width)
            parts.hang((cable,), (ends[0], ends[1]), swing, tension)
    radius = rng.uniform(0.05, 0.13)
    for a in poles:
        parts.member(plan.frame.post(float(a), float(heights.max()) + 0.4), radius, 12)
    return parts


def _fence_frame(rng: np.random.Generator, plan: _Plan, length: float, height: float) -> _Parts:
    parts = _Parts()
    spacing = rng.uniform(2.0, 3.0)
    radius = rng.uniform(0.02, 0.045)
    for a in np.arange(-0.5 * length, 0.5 * length + 1e-6, spacing):
        parts.member(plan.frame.post(float(a), height + 0.05, -radius - plan.width), radius)
    if rng.random() < 0.7:
        parts.member(plan.frame.along(length, height + 0.05, -radius - plan.width), rng.uniform(0.015, 0.022))
    return parts


def _chain_link(rng: np.random.Generator, plan: _Plan) -> _Parts:
    half = 0.5 * rng.uniform(0.035, 0.07)
    height = rng.uniform(1.0, 2.4)
    rows = int(height / half)
    length = min(plan.length, POINT_BUDGET * half / (2 * rows + 1))
    parts = _fence_frame(rng, plan, length, height)
    wire = np.arange(int(length / half))
    vertex = np.arange(rows + 1)
    offset = ((vertex[None] - wire[:, None]) % 2) * half
    a_vertex = -0.5 * length + wire[:, None] * half + offset
    b_vertex = np.where(offset > 0, 0.6, -0.6) * plan.width
    a = np.empty((len(wire), 2 * rows + 1))
    b = np.zeros_like(a)
    a[:, ::2], b[:, ::2] = a_vertex, b_vertex
    a[:, 1::2] = 0.5 * (a_vertex[:, :-1] + a_vertex[:, 1:])
    h = 0.04 + np.arange(2 * rows + 1) * 0.5 * half
    for line in plan.frame.at(a, h[None], b):
        parts.strand(line, plan.width)
    parts.strand(plan.frame.along(length, 0.04, 0.0, 0.25), 1.5 * plan.width)
    return parts


def _welded(rng: np.random.Generator, plan: _Plan) -> _Parts:
    across, up = rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2)
    height = rng.uniform(0.8, 2.2)
    length = min(plan.length, POINT_BUDGET * across / 4.0)
    parts = _fence_frame(rng, plan, length, height)
    for a in np.arange(-0.5 * length, 0.5 * length, across):
        parts.strand(plan.frame.at([a, a], [0.03, height], 0.5 * plan.width), plan.width)
    for h in np.arange(0.03, height + 1e-6, up):
        parts.strand(plan.frame.along(length, float(h), -0.5 * plan.width, 0.25), plan.width)
    return parts


def _net(rng: np.random.Generator, plan: _Plan) -> _Parts:
    spacing = rng.uniform(3.0, 8.0)
    top, bottom = rng.uniform(1.2, 4.0), rng.uniform(0.0, 0.3)
    mesh = rng.uniform(0.02, 0.15)
    net = top - bottom
    length = float(np.clip(POINT_BUDGET * mesh * mesh / (4.0 * net), 1.0, plan.length))
    dip, billow = spacing * rng.uniform(0.01, 0.05), spacing * rng.uniform(0.0, 0.06)
    waves = rng.uniform(0.5, 3.0, 2), rng.uniform(0.0, 2.0 * np.pi, 2)
    start = -0.5 * length

    def place(a: FloatArray, v: FloatArray) -> FloatArray:
        phase = ((a - start) / spacing) % 1.0
        cord = top - 4.0 * dip * phase * (1.0 - phase)
        h = bottom + (cord - bottom) * (1.0 - v / net)
        swell = np.sin(np.pi * phase) * np.sin(np.pi * v / net)
        wobble = np.sin(waves[0][0] * a + waves[1][0]) * np.sin(waves[0][1] * v + waves[1][1])
        return plan.frame.at(a, h, billow * swell + 0.2 * mesh * wobble)

    parts = _Parts()
    for sign in (1.0, -1.0):
        for c in np.arange(-net - 0.5 * length, 0.5 * length + net, mesh):
            v = np.arange(0.0, net + 1e-6, 0.5 * mesh)
            a = c + sign * v
            keep = np.abs(a) <= 0.5 * length
            if keep.sum() >= 2:
                parts.strand(place(a[keep], v[keep]), plan.width)
    cord = np.linspace(start, -start, int(length / 0.1) + 2)
    parts.strand(place(cord, np.zeros_like(cord)), 2.5 * plan.width)
    ends = plan.frame.at([start, -start], [top, top])
    parts.hang(tuple(range(len(parts.strands))), (ends[0], ends[1]), rng.uniform(1.0, 5.0), 0.0)
    for a in np.arange(start, -start + 1e-6, spacing):
        parts.member(plan.frame.post(float(a), top + 0.1), rng.uniform(0.03, 0.06))
    return parts


def _rigging(rng: np.random.Generator, plan: _Plan) -> _Parts:
    mast = float(np.clip(plan.window[1] + rng.uniform(0.5, 4.0), 2.5, 14.0))
    root = plan.frame.at(0.0, 0.0)
    parts = _Parts()
    parts.member(plan.frame.post(0.0, mast), rng.uniform(0.04, 0.12), 12)
    swing, tension = rng.uniform(0.5, 3.0), rng.uniform(0.02, 0.1)
    for _ in range(int(rng.integers(2, 9))):
        attach = root + UP * mast * rng.uniform(0.5, 1.0)
        side = rng.choice([-1.0, 1.0])
        angle = np.radians(rng.uniform(5.0, 50.0))
        direction = side * np.cos(angle) * plan.frame.u - np.sin(angle) * plan.frame.n
        reach = float(attach[1] - root[1]) * rng.uniform(0.6, 2.0)
        anchor = root + direction * reach
        anchor[1] = plan.frame.ground.height(anchor[[0]], anchor[[2]])[0]
        span = _length(anchor - attach)
        parts.member(np.stack([anchor - 0.3 * UP, anchor + 0.25 * UP]), 0.03, 8)
        rope = _rope(parts, catenary(attach, anchor + 0.2 * UP, span * rng.uniform(0.003, 0.03), 48), plan)
        parts.hang(rope, (attach, anchor + 0.2 * UP), swing, tension)
    return parts


def _railing(rng: np.random.Generator, plan: _Plan) -> _Parts:
    height = rng.uniform(0.85, 1.1)
    sides = 4 if rng.random() < 0.5 else 10
    parts = _Parts()
    for a in np.arange(-0.5 * plan.length, 0.5 * plan.length + 1e-6, rng.uniform(1.0, 2.0)):
        parts.member(plan.frame.post(float(a), height), rng.uniform(0.02, 0.04), sides)
    parts.member(plan.frame.along(plan.length, height), rng.uniform(0.02, 0.035), sides)
    if rng.random() < 0.5:
        for h in np.linspace(0.1, height, int(rng.integers(1, 5)) + 1, endpoint=False)[1:]:
            parts.member(plan.frame.along(plan.length, float(h)), 0.5 * plan.width, sides)
        return parts
    parts.member(plan.frame.along(plan.length, 0.1), 0.5 * plan.width, sides)
    for a in np.arange(-0.5 * plan.length, 0.5 * plan.length, rng.uniform(0.1, 0.16)):
        parts.member(plan.frame.at([a, a], [0.1, height]), 0.5 * plan.width, sides)
    return parts


def _antenna(rng: np.random.Generator, plan: _Plan) -> _Parts:
    mast = float(np.clip(plan.window[1] + rng.uniform(-0.5, 3.0), 2.0, 12.0))
    root = plan.frame.at(0.0, 0.0)
    parts = _Parts()
    parts.member(plan.frame.post(0.0, mast + 0.2), rng.uniform(0.02, 0.05), 10)
    for level in range(1 if rng.random() < 0.6 else 2):
        yaw = rng.uniform(0.0, 2.0 * np.pi)
        boom = np.cos(yaw) * plan.frame.u + np.sin(yaw) * plan.frame.n
        element = np.cross(UP, boom)
        centre = root + UP * (mast - 0.6 * level)
        reach = rng.uniform(1.0, 3.0)
        parts.member(np.stack([centre - 0.2 * reach * boom, centre + 0.8 * reach * boom]), 0.012, 8)
        count = int(rng.integers(4, 15))
        longest = rng.uniform(0.3, 1.2)
        for i, s in enumerate(np.linspace(-0.15, 0.75, count) * reach):
            half = 0.5 * longest * (1.0 - 0.5 * i / count)
            parts.strand(
                np.stack([centre + s * boom - half * element, centre + s * boom + half * element]), plan.width
            )
    guy = max(0.6 * plan.width, 0.003)
    for k in range(3):
        angle = rng.uniform(0.0, 2.0 * np.pi / 3.0) + 2.0 * np.pi * k / 3.0
        direction = np.cos(angle) * plan.frame.u + np.sin(angle) * plan.frame.n
        anchor = root + direction * 0.5 * mast
        anchor[1] = plan.frame.ground.height(anchor[[0]], anchor[[2]])[0] + 0.1
        attach = root + UP * 0.8 * mast
        line = parts.strand(catenary(attach, anchor, 0.01 * mast, 32), guy)
        parts.hang((line,), (attach, anchor), 0.3, 0.05)
    return parts


def _poles(rng: np.random.Generator, plan: _Plan) -> _Parts:
    spacing = rng.uniform(0.3, 3.0)
    count = int(np.clip(plan.length / spacing, 2, 14))
    a = (np.arange(count) - 0.5 * (count - 1)) * spacing + rng.normal(0.0, 0.3 * spacing, count)
    b = rng.uniform(-0.15, 0.15, count) * plan.length
    parts = _Parts()
    for ai, bi in zip(a, b, strict=True):
        height = rng.uniform(1.0, 4.0)
        base = plan.frame.at(ai, -0.3, bi)
        lean = np.tan(np.radians(rng.uniform(0.0, 6.0))) * height
        heading = rng.uniform(0.0, 2.0 * np.pi)
        tip = base + UP * height + lean * (np.cos(heading) * plan.frame.u + np.sin(heading) * plan.frame.n)
        parts.member(np.stack([base, tip]), 0.5 * plan.width, 10)
    return parts


def _grille(rng: np.random.Generator, plan: _Plan) -> _Parts:
    width, height = rng.uniform(0.8, 2.5), rng.uniform(0.8, 2.0)
    lift = rng.uniform(0.0, 0.9)
    panels = int(np.clip(np.ceil(plan.length / width), 1, 8))
    pitch, crossbars = rng.uniform(0.08, 0.16), int(rng.integers(0, 4))
    bar = 0.5 * plan.width
    parts = _Parts()
    for p in range(panels):
        left = (p - 0.5 * panels) * width
        right = left + width
        corners = [(left, lift), (right, lift), (right, lift + height), (left, lift + height), (left, lift)]
        for (a0, h0), (a1, h1) in itertools.pairwise(corners):
            parts.member(plan.frame.at([a0, a1], [h0, h1]), 1.6 * bar, 4)
        for a in np.arange(left + pitch, right - 0.5 * pitch, pitch):
            parts.member(plan.frame.at([a, a], [lift, lift + height]), bar, 4)
        for h in np.linspace(lift, lift + height, crossbars + 2)[1:-1]:
            parts.member(plan.frame.at([left, right], [h, h], -2.0 * bar), bar, 4)
        if lift > 0.05:
            for a in (left + 0.05, right - 0.05):
                parts.member(plan.frame.post(a, lift), 1.6 * bar, 4)
    return parts


def _screen(rng: np.random.Generator, plan: _Plan) -> _Parts:
    pitch = rng.uniform(0.0015, 0.006)
    height = min(rng.uniform(0.5, 1.6), 2000 * pitch)
    width = min(rng.uniform(0.5, 1.6), 2000 * pitch, plan.length)
    lift = float(np.clip(rng.uniform(*plan.window) - 0.5 * height, 0.0, 3.0))
    parts = _Parts()
    for a in np.arange(-0.5 * width, 0.5 * width, pitch):
        parts.strand(plan.frame.at([a, a], [lift, lift + height], 0.5 * plan.width), plan.width)
    for h in np.arange(lift, lift + height, pitch):
        parts.strand(plan.frame.at([-0.5 * width, 0.5 * width], [h, h], -0.5 * plan.width), plan.width)
    edge = rng.uniform(0.01, 0.02)
    corners = [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0), (-0.5, 0.0)]
    for (a0, h0), (a1, h1) in itertools.pairwise(corners):
        parts.member(
            plan.frame.at([a0 * width, a1 * width], [lift + h0 * height, lift + h1 * height]), edge, 4
        )
    if lift > 0.05:
        for a in (-0.4 * width, 0.4 * width):
            parts.member(plan.frame.post(a, lift), edge, 4)
    return parts


Builder = Callable[[np.random.Generator, _Plan], _Parts]


@dataclass(frozen=True)
class Kind:
    """width_m bounds the main thin element's diameter; length_m the structure's run."""

    build: Builder
    width_m: tuple[float, float]
    length_m: tuple[float, float]
    bands: tuple[str, ...]
    strand_finishes: tuple[str, ...]
    member_finishes: tuple[str, ...]


KINDS: dict[str, Kind] = {
    "wires": Kind(
        _wires, (0.003, 0.035), (20.0, 120.0), BANDS[1:], ("black_coat", "copper", "dark_steel"), METALS
    ),
    "chain_link": Kind(_chain_link, (0.002, 0.005), (3.0, 30.0), BANDS, COATED, METALS),
    "welded_mesh": Kind(_welded, (0.002, 0.007), (3.0, 30.0), BANDS, COATED, METALS),
    "net": Kind(
        _net, (0.001, 0.006), (3.0, 25.0), BANDS, ("natural_fibre", "orange_fibre", "black_coat"), METALS
    ),
    "rigging": Kind(
        _rigging, (0.003, 0.04), (1.0, 1.0), BANDS, ("natural_fibre", "orange_fibre"), ("weathered_wood",)
    ),
    "railing": Kind(_railing, (0.006, 0.03), (2.0, 25.0), BANDS, PAINTED, PAINTED),
    "antenna": Kind(_antenna, (0.003, 0.016), (1.0, 1.0), BANDS[1:], METALS, METALS),
    "poles": Kind(_poles, (0.004, 0.07), (0.5, 30.0), BANDS, METALS, ("weathered_wood", *PAINTED)),
    "grille": Kind(_grille, (0.004, 0.02), (1.0, 8.0), BANDS, PAINTED, PAINTED),
    "screen": Kind(_screen, (0.00025, 0.0008), (0.5, 1.6), BANDS[:2], COATED, PAINTED),
}


def _pick(rng: np.random.Generator, options: tuple[str, ...] | list[str]) -> str:
    return options[int(rng.integers(0, len(options)))]


def _band(band: str, subject: float, radius: float) -> tuple[float, float]:
    """Camera distances of a band: up to the subject's near side, or beyond its far side."""
    front = subject - radius - 0.3
    far = subject + radius + 0.5
    return {
        "foreground": (0.3, min(0.3 * subject, front)),
        "between": (max(0.35 * subject, 0.6), front),
        "background": (max(1.3 * subject, far), 5.0 * subject + far),
    }[band]


def _distance(
    rng: np.random.Generator, band: tuple[float, float], width: float, pixel_rad: float
) -> tuple[float, float]:
    """A distance and target pixel width: where the band allows, the distance that draws width at it."""
    target = float(np.exp(rng.uniform(*np.log(PIXEL_WIDTH_RANGE))))
    spread = float(np.exp(rng.uniform(np.log(band[0]), np.log(band[1]))))
    wanted = width / (target * pixel_rad)
    return (wanted if band[0] <= wanted <= band[1] else spread), target


def _horizontal(vector: FloatArray) -> FloatArray:
    return _unit(np.array([vector[0], 0.0, vector[2]]))


def _frame(
    rng: np.random.Generator, ground: Ground, camera_to_world: FloatArray, view: View, distance: float
) -> tuple[_Frame, FloatArray]:
    eye, forward = camera_to_world[:3, 3], -camera_to_world[:3, 2]
    ahead = _horizontal(forward)
    right: FloatArray = np.cross(ahead, UP)
    origin = eye + ahead * distance + right * rng.uniform(-0.4, 0.4) * distance * view.tan_half[0]
    yaw = float(np.radians(rng.uniform(-40.0, 40.0)))
    u: FloatArray = np.cos(yaw) * right + np.sin(yaw) * ahead
    across: FloatArray = np.cross(u, UP)
    frame = _Frame(ground, origin, u, across)
    centre = eye[1] + forward[1] / _length(forward[[0, 2]]) * distance
    floor = float(ground.height(origin[[0]], origin[[2]])[0])
    spread = distance * view.tan_half[1]
    return frame, np.array([centre - spread - floor, centre + spread - floor])


def _clear(points: FloatArray, view: View, anchor: int, distance: float) -> bool:
    if KDTree(points).query(view.eyes)[0].min() < max(0.2, 0.6 * distance):
        return False
    top = float(view.subject[:, 1].max()) + view.subject_height_m + 0.3
    low = points[points[:, 1] < top][:, [0, 2]]
    if len(low) and KDTree(low).query(view.subject[:, [0, 2]])[0].min() < view.subject_radius_m + 0.15:
        return False
    local = (np.linalg.inv(view.cameras[anchor]) @ np.c_[points, np.ones(len(points))].T).T
    depth = -local[:, 2]
    inside = (
        (depth > 0.05)
        & (np.abs(local[:, 0]) < depth * view.tan_half[0])
        & (np.abs(local[:, 1]) < depth * view.tan_half[1])
    )
    return int(inside.sum()) >= min(20, len(points))


def _swings(rng: np.random.Generator, hangs: list[_Hang], duration_s: float) -> tuple[Swing, ...]:
    wind = timeline.curve(rng, duration_s, WIND_PERIOD_S, 0.3, 1.0)
    tension = timeline.curve(rng, duration_s, TENSION_PERIOD_S, -1.0, 1.0)
    return tuple(
        Swing(
            h.strands,
            h.start,
            h.end,
            h.swing_rad,
            h.sag_change,
            timeline.curve(rng, duration_s, GUST_PERIOD_S, -1.0, 1.0),
            wind,
            tension,
        )
        for h in hangs
    )


def _attempt(
    rng: np.random.Generator, ground: Ground, view: View, anchor: int, *, name: str, band: str
) -> Structure | None:
    kind = KINDS[name]
    camera_to_world = view.cameras[anchor]
    subject = _length((view.subject[anchor] - camera_to_world[:3, 3])[[0, 2]])
    near, far = _band(band, subject, view.subject_radius_m)
    if near >= far:
        return None
    width = float(np.exp(rng.uniform(*np.log(kind.width_m))))
    distance, target = _distance(rng, (near, far), width, view.pixel_rad)
    frame, window = _frame(rng, ground, camera_to_world, view, distance)
    visible = 2.0 * distance * view.tan_half[0]
    length = float(np.clip(visible * rng.uniform(1.2, 2.5), *kind.length_m))
    footprint = distance * view.pixel_rad
    width = float(np.clip(target * footprint, *kind.width_m))
    low = max(float(window[0]), 0.05)
    plan = _Plan(frame, length, width, width / footprint, (low, max(float(window[1]), low + 0.1)))
    parts = kind.build(rng, plan)
    thickest = max(MIN_MEMBER_RADIUS_M, 0.5 * MAX_MEMBER_PX * footprint)
    members = tuple(Member(m.line, min(m.radius_m, thickest), m.sides) for m in parts.members)
    structure = Structure(
        name,
        band,
        distance,
        length,
        width,
        plan.width_px,
        tuple(parts.strands),
        tuple(parts.widths),
        members,
        FINISHES[_pick(rng, kind.strand_finishes)],
        FINISHES[_pick(rng, kind.member_finishes)],
        anchor,
        _swings(rng, parts.hangs, len(view.cameras) / timeline.FPS),
    )
    return structure if _clear(structure.points(), view, anchor, distance) else None


def _structure(
    rng: np.random.Generator, ground: Ground, view: View, choice: Choice, window: tuple[int, int]
) -> Structure | None:
    name = _pick(rng, sorted(KINDS))
    name = name if choice.kind is None else choice.kind
    band = _pick(rng, KINDS[name].bands)
    band = band if choice.band is None else choice.band
    if band not in KINDS[name].bands:
        raise ValueError(f"{name} is not placed in the {band} band")
    anchor = int(rng.integers(*window))
    for _ in range(ATTEMPTS):
        found = _attempt(rng, ground, view, anchor, name=name, band=band)
        if found is not None:
            return found
    return None


def sample(
    rng: np.random.Generator, ground: Ground, view: View, choices: tuple[Choice, ...] | None
) -> list[Structure]:
    """Structures for a shot; each has its own stream so fixing one never shifts another.

    Drawn structures come per WINDOW_S of shot, anchored inside their window; fixed ones anywhere.
    """
    samples = len(view.cameras)
    edges = np.linspace(0, samples, max(1, round(samples / (WINDOW_S * timeline.FPS))) + 1).astype(np.int64)
    counts = rng.choice(len(COUNT_WEIGHTS), size=len(edges) - 1, p=COUNT_WEIGHTS)
    entries = [
        (Choice(), (int(lo), int(hi)))
        for lo, hi, n in zip(edges[:-1], edges[1:], counts, strict=True)
        for _ in range(n)
    ]
    if choices is not None:
        entries = [(c, (0, samples)) for c in choices]
    streams = rng.spawn(len(entries))
    built = [_structure(s, ground, view, c, w) for s, (c, w) in zip(streams, entries, strict=True)]
    return [s for s in built if s is not None]


def _tube(member: Member) -> tuple[FloatArray, list[IntArray]]:
    line, sides = member.line, member.sides
    tangent: FloatArray = np.gradient(line, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    chord = _unit(line[-1] - line[0])
    reference = UP if abs(float(chord @ UP)) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(tangent, reference)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    side = np.cross(tangent, normal)
    angle = 2.0 * np.pi * (np.arange(sides) + 0.5) / sides
    ring = line[:, None] + member.radius_m * (
        np.cos(angle)[None, :, None] * normal[:, None] + np.sin(angle)[None, :, None] * side[:, None]
    )
    k = np.arange(sides)
    faces: list[IntArray] = [
        np.stack(
            [
                i * sides + k,
                i * sides + (k + 1) % sides,
                (i + 1) * sides + (k + 1) % sides,
                (i + 1) * sides + k,
            ],
            -1,
        )
        for i in range(len(line) - 1)
    ]
    end = (len(line) - 1) * sides
    faces += [k[::-1][None], (end + k)[None]]
    return ring.reshape(-1, 3), faces


def _author_curves(
    stage: Usd.Stage, path: str, strands: list[FloatArray], widths: list[float]
) -> UsdGeom.BasisCurves:
    curves = UsdGeom.BasisCurves.Define(stage, path)
    counts = np.array([len(s) for s in strands])
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr(ints(counts))
    curves.CreatePointsAttr(vec3f(np.concatenate(strands)))
    curves.CreateWidthsAttr(floats(np.repeat(np.asarray(widths), counts)))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curves.CreateExtentAttr(
        UsdGeom.Curves.ComputeExtent(curves.GetPointsAttr().Get(), curves.GetWidthsAttr().Get())
    )
    return curves


def _strand_groups(structure: Structure) -> list[tuple[str, list[int], Swing | None]]:
    swinging = {i for w in structure.swings for i in w.strands}
    still = [i for i in range(len(structure.strands)) if i not in swinging]
    groups: list[tuple[str, list[int], Swing | None]] = [("still", still, None)] if still else []
    return groups + [(f"span_{k}", list(w.strands), w) for k, w in enumerate(structure.swings)]


def _author_strands(
    stage: Usd.Stage,
    path: str,
    structure: Structure,
    instance_id: int,
    *,
    first_frame: int,
    samples: int,
) -> list[str]:
    UsdGeom.Xform.Define(stage, path)
    prims: list[str] = []
    seconds = timeline.sample_times(samples)
    for name, indices, swing in _strand_groups(structure):
        curves = _author_curves(
            stage,
            f"{path}/{name}",
            [structure.strands[i] for i in indices],
            [structure.strand_widths[i] for i in indices],
        )
        tag_instances(curves.GetPrim(), np.array([instance_id]))
        prims.append(str(curves.GetPath()))
        if swing is None:
            continue
        op = curves.AddTransformOp()
        for index, pose in enumerate(swing.matrices(seconds)):
            op.Set(matrix(pose.T), Usd.TimeCode(first_frame - 1 + index))
    return prims


def _author_members(stage: Usd.Stage, path: str, structure: Structure) -> UsdGeom.Mesh:
    points: list[FloatArray] = []
    faces: list[IntArray] = []
    offset = 0
    for member in structure.members:
        ring, polygons = _tube(member)
        points.append(ring)
        faces += [p + offset for p in polygons]
        offset += len(ring)
    mesh = UsdGeom.Mesh.Define(stage, path)
    merged = np.concatenate(points)
    mesh.CreatePointsAttr(vec3f(merged))
    mesh.CreateFaceVertexCountsAttr(ints(np.concatenate([np.full(len(f), f.shape[1]) for f in faces])))
    mesh.CreateFaceVertexIndicesAttr(ints(np.concatenate([f.ravel() for f in faces])))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(defaultValue=True)
    mesh.CreateExtentAttr([tuple(float(v) for v in merged.min(0)), tuple(float(v) for v in merged.max(0))])
    return mesh


@dataclass(frozen=True)
class Part:
    """An authored part of a structure: its name, instance id and the geometry prims that carry it."""

    name: str
    instance_id: int
    prims: tuple[str, ...]


def author(
    stage: Usd.Stage,
    path: str,
    structure: Structure,
    first_id: int,
    *,
    first_frame: int,
    samples: int,
) -> list[Part]:
    """Author strands and members under path, each part tagged with its own instance id.

    Swinging spans are separate curves whose transform sample i is authored at first_frame - 1 + i.
    """
    UsdGeom.Xform.Define(stage, path)
    parts: list[Part] = []
    if structure.strands:
        prims = _author_strands(
            stage, f"{path}/strands", structure, first_id, first_frame=first_frame, samples=samples
        )
        parts.append(Part("strands", first_id, tuple(prims)))
    if structure.members:
        mesh = _author_members(stage, f"{path}/members", structure)
        tag_instances(mesh.GetPrim(), np.array([first_id + len(parts)]))
        parts.append(Part("members", first_id + len(parts), (str(mesh.GetPath()),)))
    return parts
