from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d

from strata_training.augment.noise import PERMUTATION, noise_node
from strata_training.types import Array


@dataclass(frozen=True)
class CurveKey:
    x: float
    y: float
    slope: float | None = None
    right_length: float = 1.0


class NukeCurve:
    def __init__(self, keys: tuple[CurveKey, ...], steps: int = 4096) -> None:
        xs = np.array([k.x for k in keys])
        ys = np.array([k.y for k in keys])
        width, chord = np.diff(xs), np.diff(ys) / np.diff(xs)
        slopes = [k.slope for k in keys]
        for i in range(1, len(keys) - 1):
            if slopes[i] is None:
                slopes[i] = (width[i] * chord[i - 1] + width[i - 1] * chord[i]) / (
                    width[i - 1] + width[i]
                )
        if slopes[0] is None:
            slopes[0] = (3 * chord[0] - _known(slopes[1])) / 2
        if slopes[-1] is None:
            slopes[-1] = (3 * chord[-1] - _known(slopes[-2])) / 2
        self.slopes = np.array([_known(s) for s in slopes])
        t = np.linspace(0, 1, steps)[:, None]
        basis = np.hstack(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3))
        segments: list[Array] = []
        for i in range(len(keys) - 1):
            a = width[i] * keys[i].right_length / 3
            b = width[i] / 3
            control = np.array(
                [
                    [xs[i], ys[i]],
                    [xs[i] + a, ys[i] + a * self.slopes[i]],
                    [xs[i + 1] - b, ys[i + 1] - b * self.slopes[i + 1]],
                    [xs[i + 1], ys[i + 1]],
                ]
            )
            segments.append(basis @ control)
        table = np.concatenate(segments)
        self.table_x, self.table_y = table[:, 0], table[:, 1]
        self.xs, self.ys = xs, ys

    def __call__(self, value: Array) -> Array:
        inside = np.interp(value, self.table_x, self.table_y)
        below = self.ys[0] + (value - self.xs[0]) * self.slopes[0]
        above = self.ys[-1] + (value - self.xs[-1]) * self.slopes[-1]
        return np.where(value < self.xs[0], below, np.where(value > self.xs[-1], above, inside))


def _known(value: float | None) -> float:
    if value is None:
        raise ValueError("Nuke curve slopes need at least one neighbouring explicit slope")
    return value


FILM_RESPONSE = (
    NukeCurve(
        (
            CurveKey(0.002999480115, 0.1362445652),
            CurveKey(0.1373360008, 0.4889034033, 2.0, 0.7107599974),
            CurveKey(1.0, 1.239727497, 0.4499999881),
        )
    ),
    NukeCurve(
        (
            CurveKey(0.0, 0.110607639),
            CurveKey(0.135157004, 0.5191542506, 1.5),
            CurveKey(1.0, 1.144214392, 0.400000006),
        )
    ),
    NukeCurve(
        (
            CurveKey(0.0, 0.1032338738),
            CurveKey(0.1281179935, 0.3248756528),
            CurveKey(1.0, 1.0),
        )
    ),
)

ISO_STOPS = {
    160: -2.333333333,
    200: -2.0,
    250: -1.6,
    320: -1.3,
    400: -1.0,
    500: -0.6,
    640: -0.3,
    800: 0.0,
    1000: 0.3,
    1280: 0.6,
    1600: 1.0,
    2000: 1.3,
    2560: 1.6,
    3200: 2.0,
}


@dataclass(frozen=True)
class GrainProfile:
    sizes: tuple[float, float, float, float]
    amplitudes: tuple[float, float, float, float]
    channel_gain: tuple[float, float, float]
    sharpen: tuple[float, float, float]


NOISE_BASE = (0.035, 0.03, 0.01, 0.022)
NOISE_GAMMA = (0.998, 1.0, 0.998, 1.0)
NOISE_ZOFFSET = (0.0, 0.0, 10.0, 100.0)
NOISE_CHANNEL = (None, 0, 1, 2)
SHARPEN_BASE = (0.25, 0.25, 0.1)


def _profile(
    sizes: tuple[float, float, float, float],
    scale: tuple[float, float, float, float],
    gain: tuple[float, float, float],
    sharpen: tuple[float, float, float],
) -> GrainProfile:
    amplitudes = tuple(b * s for b, s in zip(NOISE_BASE, scale, strict=True))
    return GrainProfile(
        sizes,
        (amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3]),
        gain,
        (sharpen[0] * SHARPEN_BASE[0], sharpen[1] * SHARPEN_BASE[1], sharpen[2] * SHARPEN_BASE[2]),
    )


PROFILES = {
    "alexa35": _profile(
        (1.55, 4.8, 3.0, 6.0), (0.78, 0.72, 0.62, 0.70), (1.25, 1.45, 2.05), (0.85, 0.85, 0.8)
    ),
    "alexa_lf": _profile(
        (2.0, 6.5, 4.0, 8.0), (0.95, 0.92, 0.86, 0.90), (1.55, 1.9, 2.75), (1.0, 1.0, 1.0)
    ),
    "alexa_sxt": _profile(
        (2.0, 6.5, 4.0, 8.0), (1.0, 1.0, 1.0, 1.0), (1.6, 2.0, 3.0), (1.0, 1.0, 1.0)
    ),
    "alexa65": _profile(
        (2.2, 7.2, 4.5, 9.0), (0.88, 0.84, 0.78, 0.82), (1.45, 1.75, 2.45), (0.95, 0.95, 0.92)
    ),
}

SHARPEN_SIGMA = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_RADIUS = 4
_TAPS = np.exp(-0.5 * (np.arange(-_RADIUS, _RADIUS + 1) / SHARPEN_SIGMA) ** 2)
SHARPEN_KERNEL = _TAPS / _TAPS.sum()


@dataclass(frozen=True)
class GrainParams:
    profile: str
    iso: int
    exposure_offset: float
    seed: int
    amount: float


def exposure_level(offset: float) -> float:
    return float(np.clip(2.0 ** (offset * 0.5), 0.45, 3.25))


def response(image: Array, iso: int) -> Array:
    stops = ISO_STOPS[iso]
    return np.stack(
        [2.0**stops * curve(image[c] * 2.0**-stops) for c, curve in enumerate(FILM_RESPONSE)]
    ).astype(np.float32)


def sharpen(channel: Array, amount: float) -> Array:
    blurred = correlate1d(channel, SHARPEN_KERNEL, axis=0, mode="nearest")
    blurred = correlate1d(blurred, SHARPEN_KERNEL, axis=1, mode="nearest")
    return channel + amount * (channel - blurred)


def grain_noise(shape: tuple[int, int], params: GrainParams, table: Array = PERMUTATION) -> Array:
    profile = PROFILES[params.profile]
    level = exposure_level(params.exposure_offset)
    noise = np.zeros((3, *shape), dtype=np.float32)
    for size, amplitude, gamma, zoffset, channel in zip(
        profile.sizes, profile.amplitudes, NOISE_GAMMA, NOISE_ZOFFSET, NOISE_CHANNEL, strict=True
    ):
        scale = np.float32(amplitude * level)
        field = noise_node(shape, size, params.seed + zoffset, gamma=gamma, table=table) - 0.5
        if channel is None:
            noise += field * scale
        else:
            noise[channel] += field * scale
    for c in range(3):
        noise[c] = sharpen(noise[c] * np.float32(profile.channel_gain[c]), profile.sharpen[c])
    return noise * np.float32(params.amount)


def apply_grain(image: Array, params: GrainParams, table: Array = PERMUTATION) -> Array:
    noise = grain_noise(image.shape[-2:], params, table)
    return (image + noise * response(image, params.iso)).astype(np.float32)
