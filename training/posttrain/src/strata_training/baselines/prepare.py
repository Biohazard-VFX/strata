from pathlib import Path
from typing import Any

import numpy as np

from strata_training.baselines.curves import fit_curve
from strata_training.baselines.deep import analytic
from strata_training.baselines.probe import predict_probe
from strata_training.eval.protocol import entries_for
from strata_training.io import save_arrays, write_json
from strata_training.types import Array


def prepare_baselines(
    index: Path,
    protocol: Path,
    features: Path,
    output: Path,
    probe: Path,
    normalization: Path,
    calibration: Path | None = None,
) -> None:
    train = entries_for(index, protocol, "train")
    test = entries_for(index, protocol, "test")
    train_ids = {e["id"] for e in train}
    if calibration is None and not train:
        raise ValueError("Baseline curve fitting requires training frames or frozen calibration")
    output.mkdir(parents=True, exist_ok=True)
    predicted: dict[int, list[Array]] = {6: [], 16: []}
    truth: list[Array] = []
    masks: list[Array] = []
    for entry in train + test:
        target = output / entry["id"]
        target.mkdir(exist_ok=True)
        data = np.load(entry["cache"])
        base = np.load(features / entry["id"] / "base.npz")
        save_arrays(
            target / "base.npz",
            inverse_depth=base["inverse_depth"],
            raw_inverse_depth=base["raw_inverse_depth"],
            ratio=base["raw_metric_scale"] / base["metric_scale"],
        )
        for radius in (6, 16):
            path = target / f"r{radius}.npz"
            if not path.exists():
                a, f, b = analytic(base["inverse_depth"], radius)
                save_arrays(path, alpha=a, inv_fg=f, inv_bg=b)
            if entry["id"] in train_ids:
                predicted[radius].append(np.load(path)["alpha"])
        if entry["id"] in train_ids:
            truth.append(data["reference_a_fg"] if "reference_a_fg" in data else data["alpha"][0])
            masks.append(data["mixed"])
        elif not (target / "probe.npz").exists():
            prediction = predict_probe(
                probe,
                normalization,
                data["srgb"],
                base["inverse_depth"],
                np.load(features / entry["id"] / "features.npy", mmap_mode="r"),
                data["beauty"].astype(np.float32),
            )
            save_arrays(
                target / "probe.npz",
                alpha=prediction.alpha[0],
                inv_fg=prediction.inverse_depth[0],
                inv_bg=prediction.inverse_depth[1],
            )
    if calibration is not None:
        import json

        frozen: dict[str, Any] = json.loads(calibration.read_text())
    else:
        frozen = {
            "curves": {str(r): fit_curve(predicted[r], truth, masks).tolist() for r in (6, 16)},
            "train_ids": [e["id"] for e in train],
        }
    write_json(output / "curves.json", frozen)
