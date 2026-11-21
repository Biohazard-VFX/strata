import torch
from torch import nn

from strata_training.losses.deep import deep_loss
from strata_training.model.head import DeepHead
from strata_training.torch_api import api


def test_head_preserves_base_initialization_and_orders_every_layer() -> None:
    head = DeepHead(8, nn.Conv2d(8, 1, 1), layers=4)
    features = torch.randn(2, 8, 8, 8)
    image = torch.rand(2, 3, 8, 8)
    base = torch.full((2, 1, 8, 8), 0.5)
    result = head(features, image, base)
    torch.testing.assert_close(result["inverse_depth"][:, :1], base)
    assert torch.all(result["inverse_depth"][:, :-1] > result["inverse_depth"][:, 1:])
    assert torch.all((result["alpha"] > 0) & (result["alpha"] < 1))


def test_supervision_masks_invalid_depth_but_penalizes_extra_coverage() -> None:
    alpha = torch.tensor([[[[0.5]], [[1.0]], [[0.0]], [[0.0]]]])
    inverse = torch.tensor([[[[0.5]], [[0.25]], [[0.0]], [[0.0]]]])
    rgb = alpha[:, :, None].expand(-1, -1, 3, -1, -1) * 0.5
    prediction = {
        "inverse_depth": inverse.clamp_min(0.01).requires_grad_(),
        "alpha": alpha.clone().requires_grad_(),
        "rgb": rgb.clone().requires_grad_(),
    }
    target = {
        "inverse_depth": inverse,
        "alpha": alpha,
        "rgb": rgb,
        "beauty": torch.full((1, 3, 1, 1), 0.5),
        "mixed": torch.ones((1, 1, 1), dtype=torch.bool),
    }
    loss, terms = deep_loss(prediction, target)
    assert terms["depth"].item() == 0
    assert loss.item() == 0
    api.autograd.backward(loss)
    assert prediction["alpha"].grad is not None
    bad = dict(prediction)
    bad["alpha"] = alpha + torch.tensor([[[[0.0]], [[0.0]], [[0.5]], [[0.0]]]])
    assert deep_loss(bad, target)[0].item() > 0
