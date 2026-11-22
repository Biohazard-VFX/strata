from pathlib import Path

import numpy as np
import torch
from numpy.testing import assert_allclose
from torch import nn
from torch.nn import functional as functional

from strata_training.data.dataset import Frame
from strata_training.eval.predict import predict
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


def test_decoder_oracle_aligns_current_scalar_before_nonlinear_head(tmp_path: Path) -> None:
    alpha = np.zeros((4, 32, 32), dtype=np.float32)
    inverse = alpha.copy()
    alpha[0] = 1
    inverse[0] = 0.25
    cache = tmp_path / "target.npz"
    save_arrays(
        cache,
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
        raw_inverse_depth=np.full((32, 32), 2.0),
        raw_metric_scale=1.0,
        inverse_depth=np.full((32, 32), 0.125),
        metric_scale=0.0625,
    )
    entry = {"id": "frame", "cache": str(cache), "legacy_data": None}
    model = ScalarDecoder()
    raw = predict(
        model, {"mode": "decoder", "scale_mode": "raw", "layers": 4}, Frame(entry, root, "raw")
    )
    oracle = predict(
        model,
        {"mode": "decoder", "scale_mode": "oracle-aligned", "layers": 4},
        Frame(entry, root, "oracle-aligned"),
    )
    assert_allclose(raw.inverse_depth[0], 2)
    assert_allclose(oracle.inverse_depth[0], 0.25)
    assert (raw.alpha[0] - oracle.alpha[0]).mean() > 0.2
