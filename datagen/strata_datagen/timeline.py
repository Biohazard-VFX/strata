"""Smooth seeded signals over a shot's duration: every time-varying parameter is one of these."""

from dataclasses import dataclass

import numpy as np

from strata_datagen.usdutil import FloatArray

FPS = 24.0


@dataclass(frozen=True)
class Curve:
    """A uniform cubic B-spline through seeded keys spaced period_s apart.

    It is C2-continuous, stays inside [min(keys), max(keys)], and its slope is at most the largest
    difference between neighbouring keys divided by period_s, so per-frame steps are bounded.
    """

    keys: FloatArray
    period_s: float

    def __call__(self, t_s: FloatArray) -> FloatArray:
        u = np.clip(np.asarray(t_s, np.float64) / self.period_s, 0.0, len(self.keys) - 3 - 1e-9)
        i = np.floor(u).astype(np.int64)
        f = u - i
        w = np.stack([(1 - f) ** 3, 3 * f**3 - 6 * f**2 + 4, -3 * f**3 + 3 * f**2 + 3 * f + 1, f**3], axis=-1)
        return (w * self.keys[i[..., None] + np.arange(4)]).sum(axis=-1) / 6.0

    @property
    def max_slope(self) -> float:
        return float(np.abs(np.diff(self.keys)).max()) / self.period_s


def curve(rng: np.random.Generator, duration_s: float, period_s: float, low: float, high: float) -> Curve:
    """Keys drawn uniformly in [low, high]; enough keys to cover duration_s."""
    count = int(np.ceil(duration_s / period_s)) + 4
    return Curve(rng.uniform(low, high, count), period_s)


def sample_times(count: int) -> FloatArray:
    """Seconds of `count` consecutive frame samples starting at 0."""
    return np.arange(count) / FPS


def max_step(values: FloatArray) -> float:
    """Largest change between consecutive samples along the first axis."""
    steps = np.abs(np.diff(np.asarray(values, np.float64), axis=0))
    return float(steps.max()) if steps.size else 0.0
