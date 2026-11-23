import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch import nn
from torch.nn import functional as functional

from strata_training.config import training_defaults, validate_config
from strata_training.data.cache import CACHE_VERSION
from strata_training.data.dataset import Frame
from strata_training.data.feature_cache import base_arrays, read_base
from strata_training.eval.predict import cache_predictions, predict
from strata_training.io import save_array, save_arrays
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead


class ScalarDecoder(DeepDepthPro):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        inverse = nn.Conv2d(1, 1, 1)
        self.deep_head = DeepHead(1, inverse)
        with torch.no_grad():
            for parameter in self.deep_head.parameters():
                parameter.zero_()
            assert inverse.bias is not None
            inverse.bias.fill_(2)
            self.deep_head.stem.weight[0, -1] = 1
            self.deep_head.output.weight.reshape(4, 5, 48)[:, 1, 0] = 1

    def decode(self, pyramid: list[torch.Tensor]) -> torch.Tensor:
        return functional.interpolate(pyramid[0], scale_factor=4, mode="nearest")


def test_decoder_preserves_its_metric_prior_despite_gt_scale(
    tmp_path: Path, math_device: str
) -> None:
    alpha = np.zeros((4, 32, 32), dtype=np.float32)
    inverse = alpha.copy()
    alpha[0] = 1
    inverse[0] = 0.25
    cache = tmp_path / "target.npz"
    save_arrays(
        cache,
        version=np.array(CACHE_VERSION),
        alpha=alpha,
        inverse_depth=inverse,
        rgb=np.zeros((4, 3, 32, 32)),
        beauty=np.zeros((3, 32, 32)),
        mixed=np.zeros((32, 32), dtype=bool),
    )
    root = tmp_path / "features"
    save_array(root / "frame/features.npy", np.zeros((1, 32, 32)))
    for level, size in enumerate((8, 4, 2, 1)):
        save_array(root / f"frame/pyramid_{level}.npy", np.zeros((1, size, size)))
    save_arrays(
        root / "frame/base.npz",
        **base_arrays(np.full((32, 32), 2.0), 0.5),
    )
    entry = {"id": "frame", "cache": str(cache)}
    model = ScalarDecoder().to(math_device)
    raw = predict(model, {"mode": "decoder", "layers": 4}, Frame(entry, root))
    assert_allclose(raw.inverse_depth[0], 1)
    second = tmp_path / "second.npz"
    save_arrays(
        second,
        version=np.array(CACHE_VERSION),
        alpha=alpha,
        inverse_depth=inverse * 8,
        rgb=np.zeros((4, 3, 32, 32)),
        beauty=np.zeros((3, 32, 32)),
        mixed=np.zeros((32, 32), dtype=bool),
    )
    changed = predict(
        model, {"mode": "decoder", "layers": 4}, Frame({**entry, "cache": str(second)}, root)
    )
    for key, value in raw.arrays().items():
        np.testing.assert_array_equal(value, changed.arrays()[key])


@pytest.mark.parametrize("mode", ["cached", "hybrid"])
@pytest.mark.parametrize("key", ["scale_mode", "scale", "depth_scale"])
@pytest.mark.parametrize("value", ["raw", "oracle-aligned", None])
def test_training_rejects_any_scale_configuration(mode: str, key: str, value: Any) -> None:
    config = training_defaults(
        {
            "mode": "head",
            "input_mode": mode,
            "samples": "samples",
            "learning_rate": 4e-4,
            "steps": 100,
            "batch_size": 2,
        }
    )
    validate_config({**config, "grain_scale": 0.5})
    with pytest.raises(ValueError, match="Scale configuration is forbidden"):
        validate_config({**config, key: value})


def test_prediction_api_has_no_scale_override() -> None:
    for function in (cache_predictions, Frame):
        assert not any("scale" in name for name in inspect.signature(function).parameters)


@pytest.mark.parametrize("version", [None, "features-metric-v1", "duplicate"])
def test_unversioned_or_stale_feature_caches_are_rejected(
    tmp_path: Path, version: str | None
) -> None:
    path = tmp_path / "base.npz"
    values = base_arrays(np.ones((2, 2)), 1.0)
    if version is None:
        del values["version"]
    elif version == "duplicate":
        values["raw_inverse_depth"] = np.ones((2, 2))
    else:
        values["version"] = np.array(version)
    save_arrays(path, **values)
    with pytest.raises(ValueError, match="Feature cache version mismatch"):
        read_base(path)
