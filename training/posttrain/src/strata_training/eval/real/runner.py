import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from strata_training.eval.real.fetch import checksum
from strata_training.eval.real.metrics import matte_metrics, temporal_error
from strata_training.eval.real.readers import Sample, samples
from strata_training.io import write_json
from strata_training.types import Array


def evaluate(
    root: Path,
    output: Path,
    predictor: Callable[[Sample], Array],
    name: str,
    limit: int | None = None,
    frames_per_sequence: int | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rows: list[dict[str, Any]] = []
    previous: tuple[Sample, Array] | None = None
    for sample in samples(root, frames_per_sequence):
        prediction = predictor(sample)
        row: dict[str, Any] = {"id": sample.identifier, **matte_metrics(prediction, sample.alpha)}
        row["dtssd"] = None
        if previous is not None:
            old, old_prediction = previous
            if (
                sample.sequence is not None
                and sample.sequence == old.sequence
                and old.frame is not None
                and sample.frame == old.frame + 1
            ):
                row["dtssd"] = temporal_error(prediction, old_prediction, sample.alpha, old.alpha)
        previous = sample, prediction
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No evaluation samples")
    means: dict[str, float | None] = {}
    valid: dict[str, int] = {}
    for key in rows[0]:
        if key == "id":
            continue
        values = [r[key] for r in rows if r[key] is not None]
        means[key] = float(np.mean(values)) if values else None
        valid[key] = len(values)
    report = {
        "protocol": "real-coverage-v1-front-slot",
        "baseline": name,
        "dataset": root.name,
        "manifest_sha256": checksum(root / "manifest.json"),
        "evaluated": len(rows),
        "requested_limit": limit,
        "frames_per_sequence": frames_per_sequence,
        "macro_mean": means,
        "valid_images": valid,
        "per_image": rows,
    }
    write_json(output, report)
    return report


def cache_path(directory: Path, identifier: str) -> Path:
    return directory / (hashlib.sha256(identifier.encode()).hexdigest() + ".npz")


def cached(directory: Path) -> Callable[[Sample], Array]:
    def predict(sample: Sample) -> Array:
        with np.load(cache_path(directory, sample.identifier), allow_pickle=False) as file:
            alpha = file["alpha"]
        if alpha.ndim == 3:
            from strata_training.eval.real.protocol import deep_to_matte

            return deep_to_matte(alpha)
        return alpha

    return predict


def fake(sample: Sample) -> Array:
    return np.ones(sample.alpha.shape, dtype=np.float64)
