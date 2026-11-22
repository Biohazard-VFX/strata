import numpy as np
from scipy import ndimage

from strata_training.eval.real.protocol import validate_matte
from strata_training.types import Array


def masked_mean(value: Array, mask: Array) -> float | None:
    return float(np.mean(value[mask])) if np.any(mask) else None


def boundary_mask(gt: Array, radius: int = 5) -> Array:
    if radius < 1:
        raise ValueError("Boundary radius must be positive")
    core = gt >= 0.5
    edge = np.zeros_like(core)
    edge[1:] |= core[1:] != core[:-1]
    edge[:-1] |= core[1:] != core[:-1]
    edge[:, 1:] |= core[:, 1:] != core[:, :-1]
    edge[:, :-1] |= core[:, 1:] != core[:, :-1]
    edge |= (gt > 0) & (gt < 1)
    return ndimage.maximum_filter(edge, size=2 * radius + 1, mode="nearest")


def gradient(value: Array) -> Array:
    sigma = 1.4
    half = int(np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * 0.01))))
    x = np.arange(-half, half + 1, dtype=np.float64)
    gaussian = np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    kernel: Array = gaussian[:, None] * (-x * gaussian / sigma**2)[None, :]
    kernel /= np.sqrt(np.sum(kernel**2))
    gx: Array = ndimage.correlate(value, kernel, mode="nearest")
    gy: Array = ndimage.correlate(value, kernel.T, mode="nearest")
    return np.hypot(gx, gy)


def connectivity(pred: Array, gt: Array) -> float:
    level = np.full(gt.shape, -1.0)
    for step in range(1, 11):
        threshold = step / 10
        labels, count = ndimage.label((pred >= threshold) & (gt >= threshold))
        omega = np.zeros_like(gt, dtype=bool)
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            omega = labels == np.argmax(sizes)
        level[(level < 0) & ~omega] = (step - 1) / 10
    level[level < 0] = 1
    dp, dg = pred - level, gt - level
    phi_pred = 1 - dp * (dp >= 0.15)
    phi_gt = 1 - dg * (dg >= 0.15)
    return float(np.sum(np.abs(phi_pred - phi_gt)) / 1000)


def matte_metrics(pred: Array, gt: Array) -> dict[str, float | None]:
    pred, gt = validate_matte(pred), validate_matte(gt)
    if pred.shape != gt.shape:
        raise ValueError("Prediction and GT shape mismatch; no label resizing")
    error = np.abs(pred - gt)
    mixed = (gt > 0) & (gt < 1)
    return {
        "mixed_mae": masked_mean(error, mixed),
        "mixed_mse": masked_mean(error**2, mixed),
        "mixed_mae_8bit": masked_mean(error, (gt > 1 / 255) & (gt < 254 / 255)),
        "full_mae": float(np.mean(error)),
        "boundary_mae": masked_mean(error, boundary_mask(gt)),
        "sad": float(np.sum(error) / 1000),
        "grad": float(np.sum((gradient(pred) - gradient(gt)) ** 2) / 1000),
        "conn": connectivity(pred, gt),
        "background_alpha": masked_mean(pred, gt == 0),
        "foreground_missing_alpha": masked_mean(1 - pred, gt == 1),
        "mixed_pixels": float(np.count_nonzero(mixed)),
        "pixels": float(gt.size),
    }


def temporal_error(pred: Array, previous: Array, gt: Array, previous_gt: Array) -> float:
    values = [validate_matte(v) for v in (pred, previous, gt, previous_gt)]
    if any(v.shape != values[0].shape for v in values):
        raise ValueError("Temporal shapes differ")
    return float(100 * np.sqrt(np.mean(((values[0] - values[1]) - (values[2] - values[3])) ** 2)))
