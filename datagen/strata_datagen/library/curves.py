"""Seeded smooth functions of time: the parameters a long shot varies gradually instead of cutting.

Every curve is a sum of a few low-frequency sinusoids with seeded phases, so it is smooth (all
derivatives continuous), deterministic per seed and has no preferred start or end.
"""

import numpy as np

from strata_datagen.usdutil import FloatArray

HARMONICS = 3


def smooth(
    rng: np.random.Generator, seconds: FloatArray, count: int, period_s: tuple[float, float]
) -> FloatArray:
    """`count` curves in [-1, 1] sampled at `seconds`: (len(seconds), count)."""
    periods = rng.uniform(period_s[0], period_s[1], (count, HARMONICS))
    phases = rng.uniform(0.0, 2.0 * np.pi, (count, HARMONICS))
    amplitudes = rng.uniform(0.5, 1.0, (count, HARMONICS))
    waves = np.sin(2.0 * np.pi * seconds[:, None, None] / periods[None] + phases[None]) * amplitudes[None]
    return waves.sum(axis=-1) / amplitudes.sum(axis=-1)[None]


def mixture(
    rng: np.random.Generator, seconds: FloatArray, count: int, period_s: tuple[float, float], sharpness: float
) -> FloatArray:
    """Convex weights over `count` sources that drift smoothly: (len(seconds), count), rows sum to 1."""
    logits = sharpness * smooth(rng, seconds, count, period_s)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)
