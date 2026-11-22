"""FP16 export precision must survive residual activations beyond the float16 range."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import cast

import torch
from depth_pro_2 import DepthPro2
from strata_train._external import resolve
from strata_train._precision import keep_float32_boundaries
from strata_train.export import Depth
from torch import Tensor, nn


def seed(value: int) -> None:
    cast("Callable[[int], object]", resolve(torch, "manual_seed"))(value)


def massive_activation_model() -> DepthPro2:
    seed(0)
    model = DepthPro2(
        backbone_preset="vit_small_patch16_dinov3",
        reassembled_blocks=(2, 5, 8, 11),
        cross_frame_blocks=(5, 7, 9, 11),
        dims_reassemble=(48, 96, 192, 384),
        dim_decoder=64,
        fov=False,
        precision=torch.float16,
    ).eval()
    block = model.encoder.blocks[11]
    fc2 = cast("nn.Linear", resolve(resolve(block, "mlp"), "fc2"))
    gamma = cast("Tensor", resolve(block, "gamma_2"))
    with torch.no_grad():
        fc2.weight[7].zero_()
        fc2.bias[7] = 1
        gamma[7] = 1e7
    return model


def inverse_depth(model: DepthPro2) -> Tensor:
    seed(1)
    image = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        return Depth(model)(image)


class PrecisionTests(unittest.TestCase):
    def test_stock_fp16_tap_overflows(self) -> None:
        self.assertFalse(bool(torch.isfinite(inverse_depth(massive_activation_model())).all()))

    def test_float32_boundaries_keep_massive_residual_finite(self) -> None:
        model = massive_activation_model()
        keep_float32_boundaries(model)
        prediction = inverse_depth(model)
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(prediction).all()))


if __name__ == "__main__":
    unittest.main()
