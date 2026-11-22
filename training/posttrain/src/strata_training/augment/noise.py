import numpy as np

from strata_training.types import Array

PERIOD = 2048
GRADIENTS = np.array(
    [
        [1, 1, 0],
        [-1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [-1, 0, 1],
        [0, 1, 1],
        [0, -1, 1],
        [1, 0, -1],
        [-1, 0, -1],
        [0, 1, -1],
        [0, -1, -1],
        [1, 1, 0],
        [-1, 1, 0],
        [0, -1, 1],
        [0, -1, -1],
    ],
    dtype=np.float32,
)
_GX, _GY, _GZ = (np.ascontiguousarray(GRADIENTS[:, i]) for i in range(3))
_ANGLE = np.radians(-30.0)
_COS, _SIN = np.cos(_ANGLE), np.sin(_ANGLE)
ROTATION = np.array([[1, 0, 0], [0, _COS, -_SIN], [0, _SIN, _COS]]) @ np.array(
    [[_COS, 0, _SIN], [0, 1, 0], [-_SIN, 0, _COS]]
)


def permutation(seed: int) -> Array:
    table = np.random.default_rng(seed).permutation(PERIOD).astype(np.int32)
    return np.concatenate((table, table))


PERMUTATION = permutation(0x5EED)


def gradient_noise(x: Array, y: Array, z: Array, table: Array = PERMUTATION) -> Array:
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    ix, iy, iz = ((f.astype(np.int32) & (PERIOD - 1)) for f in (fx, fy, fz))
    x, y, z = x - fx, y - fy, z - fz
    wx = (1 - x * x * (3 - 2 * x), x * x * (3 - 2 * x))
    wy = (1 - y * y * (3 - 2 * y), y * y * (3 - 2 * y))
    w = z * z * (3 - 2 * z)
    xs, ys = (x, x - 1), (y, y - 1)
    a, b = np.take(table, ix) + iy, np.take(table, ix + 1) + iy
    total = np.zeros_like(x)
    for base, dx, dy in ((a, 0, 0), (b, 1, 0), (a + 1, 0, 1), (b + 1, 1, 1)):
        corner = np.take(table, base) + iz
        near, far = corner & 15, (corner + 1) & 15
        planar = np.take(_GX, near) * xs[dx] + np.take(_GY, near) * ys[dy] + np.take(_GZ, near) * z
        back = np.take(_GX, far) * xs[dx] + np.take(_GY, far) * ys[dy] + np.take(_GZ, far) * (z - 1)
        total += wx[dx] * wy[dy] * (planar + w * (back - planar))
    return total


def octave_count(size: float, octaves: int = 10, lacunarity: float = 2.0) -> int:
    return max(1, sum(size / lacunarity**i > 1 for i in range(octaves)))


BLOCK_PIXELS = 1 << 14


def noise_node(
    shape: tuple[int, int],
    size: float,
    zoffset: float,
    offset: tuple[float, float] = (0.0, 0.0),
    gamma: float = 1.0,
    table: Array = PERMUTATION,
) -> Array:
    height, width = shape
    octaves = octave_count(size)
    cols = np.arange(width, dtype=np.float64) / size + offset[0]
    total = np.empty(shape, dtype=np.float32)
    step = max(1, BLOCK_PIXELS // width)
    for top in range(0, height, step):
        rows = (height - 1 - np.arange(top, min(top + step, height), dtype=np.float64)) / size
        plane = np.stack(np.broadcast_arrays(cols[None, :], rows[:, None] + offset[1], zoffset))
        point = np.einsum("ij,jhw->ihw", ROTATION, plane).astype(np.float32)
        block = np.zeros(point.shape[1:], dtype=np.float32)
        for octave in range(octaves):
            block += gradient_noise(*(point * np.float32(2.0**octave)), table=table)
        total[top : top + step] = block
    value = 0.5 + 0.5 * total
    if gamma == 1:
        return value
    return np.where(value > 0, np.abs(value) ** np.float32(1 / gamma), value)
