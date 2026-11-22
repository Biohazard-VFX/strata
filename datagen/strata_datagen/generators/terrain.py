"""Noise heightfield terrain: fBm, ridged and domain-warped noise, thermal erosion, slope/height masks."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom

from strata_datagen.fields import Grid, Ground
from strata_datagen.usdutil import ints, vec2f, vec3f

VERSION = "terrain-0.2.0"


@dataclass(frozen=True)
class TerrainParams:
    size_m: float
    cell_m: float
    amplitude_m: float
    wavelength_m: float
    octaves: int
    ridged: float
    warp_m: float
    erosion_iterations: int
    talus_deg: float
    flat_radius_m: float
    edge_drop_m: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sample_params(rng: np.random.Generator) -> TerrainParams:
    return TerrainParams(
        size_m=300.0,
        cell_m=0.5,
        amplitude_m=float(rng.uniform(1.0, 9.0)),
        wavelength_m=float(rng.uniform(25.0, 90.0)),
        octaves=int(rng.integers(4, 7)),
        ridged=float(rng.uniform(0.0, 0.8)),
        warp_m=float(rng.uniform(0.0, 25.0)),
        erosion_iterations=int(rng.integers(10, 60)),
        talus_deg=float(rng.uniform(28.0, 40.0)),
        flat_radius_m=float(rng.uniform(6.0, 14.0)),
        edge_drop_m=float(rng.uniform(6.0, 20.0)),
    )


class _Perlin:
    """2D gradient noise in [-1, 1] over a 256-cell periodic lattice."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.perm = np.concatenate([rng.permutation(256)] * 2).astype(np.int64)
        angles = rng.uniform(0.0, 2.0 * np.pi, 256)
        self.grads = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def _dot(self, ix: NDArray[np.int64], iz: NDArray[np.int64], fx: Grid, fz: Grid) -> Grid:
        g = self.grads[self.perm[self.perm[ix & 255] + (iz & 255)]]
        return g[..., 0] * fx + g[..., 1] * fz

    def __call__(self, x: Grid, z: Grid) -> Grid:
        x0, z0 = np.floor(x), np.floor(z)
        fx, fz = x - x0, z - z0
        ix, iz = x0.astype(np.int64), z0.astype(np.int64)
        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        v = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
        n00 = self._dot(ix, iz, fx, fz)
        n10 = self._dot(ix + 1, iz, fx - 1, fz)
        n01 = self._dot(ix, iz + 1, fx, fz - 1)
        n11 = self._dot(ix + 1, iz + 1, fx - 1, fz - 1)
        top = n00 + u * (n10 - n00)
        return np.sqrt(2.0) * (top + v * (n01 + u * (n11 - n01) - top))


def fbm(noise: _Perlin, x: Grid, z: Grid, octaves: int, ridged: float) -> Grid:
    """Fractal sum normalised to about [-1, 1]; ridged blends in (1 - |n|)^2 octaves."""
    total = np.zeros_like(x)
    amplitude, frequency, norm = 1.0, 1.0, 0.0
    for octave in range(octaves):
        n = noise(x * frequency + 17.3 * octave, z * frequency - 9.1 * octave)
        ridge = (1.0 - np.abs(n)) ** 2 * 2.0 - 1.0
        total += amplitude * ((1.0 - ridged) * n + ridged * ridge)
        norm += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return total / norm


def thermal_erosion(heights: Grid, cell_m: float, talus_deg: float, iterations: int) -> Grid:
    """Move material downhill wherever a 4-neighbour slope exceeds the talus angle."""
    h = heights.copy()
    limit = np.tan(np.radians(talus_deg)) * cell_m
    for _ in range(iterations):
        padded = np.pad(h, 1, mode="edge")
        moved = np.zeros_like(h)
        for dz, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbour = padded[1 + dz : padded.shape[0] - 1 + dz, 1 + dx : padded.shape[1] - 1 + dx]
            excess = np.maximum(h - neighbour - limit, 0.0) * 0.125
            moved -= excess
            moved += np.roll(np.roll(excess, dz, axis=0), dx, axis=1)
        h += moved
    return h


def _smoothstep(edge0: float, edge1: float, x: Grid) -> Grid:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _shape(raw: Grid, x: Grid, z: Grid, params: TerrainParams) -> Grid:
    radius = np.hypot(x, z)
    centre = float(np.mean(raw[radius < params.flat_radius_m]))
    flat = 1.0 - _smoothstep(params.flat_radius_m, 3.0 * params.flat_radius_m, radius)
    return raw + (centre - raw) * 0.85 * flat - centre


def edge_falloff(relief: Grid, x: Grid, z: Grid, params: TerrainParams) -> Grid:
    """Flatten the relief and sink it beyond the scattered area, so bare ground drops below the horizon."""
    half = 0.5 * params.size_m
    edge = _smoothstep(0.4 * half, 0.95 * half, np.hypot(x, z))
    return relief * (1.0 - edge) - params.edge_drop_m * edge


def _masks(heights: Grid, cell_m: float, cover: Grid) -> dict[str, Grid]:
    gz, gx = np.gradient(heights, cell_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    low, high = np.percentile(heights, [5.0, 95.0])
    height01 = np.clip((heights - low) / max(high - low, 1e-6), 0.0, 1.0)
    return {
        "flat": 1.0 - _smoothstep(12.0, 32.0, slope),
        "steep": _smoothstep(24.0, 42.0, slope),
        "low": 1.0 - height01,
        "high": height01,
        "cover": cover,
    }


def build(params: TerrainParams, rng: np.random.Generator) -> Ground:
    count = round(params.size_m / params.cell_m) + 1
    axis = np.linspace(-0.5 * params.size_m, 0.5 * params.size_m, count)
    x, z = np.meshgrid(axis, axis)
    noise = _Perlin(rng)
    s = 1.0 / params.wavelength_m
    wx = params.warp_m * fbm(noise, x * s * 0.5 + 31.7, z * s * 0.5, 3, 0.0)
    wz = params.warp_m * fbm(noise, x * s * 0.5, z * s * 0.5 - 47.3, 3, 0.0)
    raw = params.amplitude_m * fbm(noise, (x + wx) * s, (z + wz) * s, params.octaves, params.ridged)
    relief = thermal_erosion(
        _shape(raw, x, z, params), params.cell_m, params.talus_deg, params.erosion_iterations
    )
    cover = 0.5 + 0.5 * fbm(noise, x * 0.08 + 101.0, z * 0.08 - 77.0, 4, 0.0)
    return Ground(
        heights=edge_falloff(relief, x, z, params),
        cell_m=params.cell_m,
        masks=_masks(relief, params.cell_m, cover),
    )


def _faces(count: int) -> NDArray[np.int64]:
    index = np.arange(count * count, dtype=np.int64).reshape(count, count)
    a, b = index[:-1, :-1], index[:-1, 1:]
    c, d = index[1:, 1:], index[1:, :-1]
    return np.stack([a, d, c, b], axis=-1).ravel()


def author(stage: Usd.Stage, path: str, ground: Ground) -> UsdGeom.Mesh:
    """A vertex-normal quad mesh with world-metre st coordinates for tiling materials."""
    count = ground.heights.shape[0]
    axis = np.linspace(-ground.half_size_m, ground.half_size_m, count)
    x, z = np.meshgrid(axis, axis)
    points = np.stack([x, ground.heights, z], axis=-1).reshape(-1, 3)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(vec3f(points))
    mesh.CreateFaceVertexCountsAttr(ints(np.full((count - 1) ** 2, 4)))
    mesh.CreateFaceVertexIndicesAttr(ints(_faces(count)))
    mesh.CreateNormalsAttr(vec3f(ground.normal(x, z).reshape(-1, 3)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(vec2f(points[:, [0, 2]]))
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    return mesh
