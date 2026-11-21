import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strata_training.baselines.curves import fit_curve
from strata_training.baselines.deep import analytic
from strata_training.baselines.probe import predict_probe
from strata_training.eval.protocol import entries_for
from strata_training.types import Array


def prepare_baselines(
    index: Path, protocol: Path, features: Path, output: Path, probe: Path, normalization: Path
) -> None:
    torch.set_num_threads(8)
    train = entries_for(index, protocol, "train")
    test = entries_for(index, protocol, "test")
    output.mkdir(parents=True, exist_ok=True)
    predicted: dict[int, list[Array]] = {6: [], 16: []}
    truth: list[Array] = []
    masks: list[Array] = []
    for entry in train + test:
        target = output / entry["id"]
        target.mkdir(exist_ok=True)
        data = np.load(entry["cache"])
        base = np.load(features / entry["id"] / "base.npz")
        np.savez_compressed(
            target / "base.npz",
            inverse_depth=base["inverse_depth"],
            raw_inverse_depth=base["raw_inverse_depth"],
            ratio=base["raw_metric_scale"] / base["metric_scale"],
        )
        for radius in (6, 16):
            path = target / f"r{radius}.npz"
            if not path.exists():
                if entry["legacy_data"]:
                    source = np.load(Path(entry["legacy_data"]) / f"analytic_r{radius}.npz")
                    a, f, b = source["alpha"], source["inv_fg"], source["inv_bg"]
                else:
                    a, f, b = analytic(base["inverse_depth"], radius)
                np.savez_compressed(path, alpha=a, inv_fg=f, inv_bg=b)
            if entry["split"] == "train":
                predicted[radius].append(np.load(path)["alpha"])
        if entry["split"] == "train":
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
            np.savez_compressed(
                target / "probe.npz",
                alpha=prediction.alpha[0],
                inv_fg=prediction.inverse_depth[0],
                inv_bg=prediction.inverse_depth[1],
            )
        print(json.dumps({"baseline_cache": entry["id"]}), flush=True)
    curves: dict[str, Any] = {
        str(r): fit_curve(predicted[r], truth, masks).tolist() for r in (6, 16)
    }
    (output / "curves.json").write_text(
        json.dumps({"curves": curves, "train_ids": [e["id"] for e in train]}, indent=2)
    )
