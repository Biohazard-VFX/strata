import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch import nn

from strata_training.baselines.cached import cached_baselines
from strata_training.baselines.prepare import prepare_baselines
from strata_training.baselines.probe import FrozenProbe
from strata_training.data import mapped
from strata_training.data.feature_cache import base_arrays
from strata_training.eval.predict import cache_predictions
from strata_training.io import save_array, save_arrays, write_json
from strata_training.model.head import DeepHead


def test_metric_inputs_and_frozen_ood_calibration_preserve_training_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "features"
    frame = root / "ood"
    frame.mkdir(parents=True)
    cache = tmp_path / "target.npz"
    shape = (16, 16)
    save_arrays(
        cache,
        beauty=np.ones((3, *shape)),
        srgb=np.full((3, *shape), 128),
        mixed=np.ones(shape, dtype=bool),
    )
    save_array(frame / "features.npy", np.zeros((128, *shape), dtype=np.float32))
    for level in range(4):
        save_array(frame / f"pyramid_{level}.npy", np.zeros((1, 1, 1)))
    save_arrays(
        frame / "base.npz",
        **base_arrays(np.full(shape, 0.5), 1.0),
    )
    index, protocol = tmp_path / "index.json", tmp_path / "protocol.json"
    write_json(index, [{"id": "ood", "cache": str(cache), "split": "train"}])
    write_json(
        protocol,
        {
            "train": [],
            "test": ["ood"],
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
    )
    config = {
        "mode": "head",
        "preset": "depthpro2-h",
        "layers": 4,
        "features": "unused",
    }
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "model": DeepHead(128, nn.Identity()).state_dict()}, checkpoint)
    raw = tmp_path / "raw"
    report = cache_predictions(index, protocol, checkpoint, raw, 0, "cpu", root)
    assert report["training_config"] == config
    assert report["config"] == {**config, "features": str(root)}
    assert torch.load(checkpoint, weights_only=True)["config"] == config
    assert_allclose(np.load(raw / "ood.npz")["inverse_depth"][0], 0.5)
    probe = tmp_path / "probe.pt"
    torch.save(FrozenProbe().state_dict(), probe)
    norm = tmp_path / "norm.npz"
    save_arrays(norm, mean=np.zeros(132), std=np.ones(132))
    frozen = {
        "curves": {str(r): np.linspace(0, 1, 20).tolist() for r in (6, 16)},
        "train_ids": ["original-training-frame"],
    }
    calibration = tmp_path / "curves.json"
    write_json(calibration, frozen)
    output = tmp_path / "baselines"
    prepare_baselines(index, protocol, root, output, probe, norm, calibration)
    assert json.loads((output / "curves.json").read_text()) == frozen
    assert (output / "ood/probe.npz").exists()
    baselines = cached_baselines("ood", output, np.ones((3, *shape)))
    assert len(baselines) == 6
    assert all(name.endswith("raw metric") for name in baselines)
    for name, prediction in baselines.items():
        if name.startswith(("fake", "analytic")):
            assert_allclose(prediction.inverse_depth[0], 0.5)
    prepare_baselines(index, protocol, root, output, probe, norm, calibration)
    mapped.arrays.cache_clear()
    repeated = cached_baselines("ood", output, np.ones((3, *shape)))
    for name, prediction in repeated.items():
        for key, value in prediction.arrays().items():
            np.testing.assert_array_equal(value, baselines[name].arrays()[key])
    changed = tmp_path / "changed"
    save_arrays(changed / "ood/base.npz", **base_arrays(np.full(shape, 0.25), 1.0))
    with pytest.raises(ValueError, match="baseline cache input changed"):
        prepare_baselines(index, protocol, changed, output, probe, norm, calibration)
    torch.save(
        {
            "config": {**config, "scale_mode": "raw"},
            "model": DeepHead(128, nn.Identity()).state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="Scale configuration is forbidden"):
        cache_predictions(index, protocol, checkpoint, tmp_path / "obsolete", 0, "cpu", root)
