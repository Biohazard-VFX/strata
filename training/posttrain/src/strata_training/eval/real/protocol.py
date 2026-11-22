import numpy as np

from strata_training.types import Array


def validate_matte(value: Array) -> Array:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("Matte must be a nonempty finite HxW array")
    if np.any((value < 0) | (value > 1)):
        raise ValueError("Matte outside range [0, 1]")
    return value


def deep_to_matte(alpha: Array, foreground: Array | None = None) -> Array:
    if alpha.ndim != 3 or alpha.shape[0] == 0:
        raise ValueError("Expected nonempty KxHxW conditional coverage")
    for layer in alpha:
        validate_matte(layer)
    if foreground is None:
        return alpha[0].astype(np.float64)
    if foreground.shape != alpha.shape or foreground.dtype != np.bool_:
        raise ValueError("Foreground membership must be boolean KxHxW")
    transmittance = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    return np.sum(transmittance * alpha * foreground, axis=0)
