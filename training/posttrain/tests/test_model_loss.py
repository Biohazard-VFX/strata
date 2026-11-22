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
        "alpha_logits": torch.logit(alpha, eps=1e-7).requires_grad_(),
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
    torch.testing.assert_close(loss, torch.tensor(0.6931473))
    api.autograd.backward(loss)
    assert prediction["alpha"].grad is not None
    bad = dict(prediction)
    bad["alpha_logits"] = torch.logit(
        alpha + torch.tensor([[[[0.0]], [[0.0]], [[0.5]], [[0.0]]]]), eps=1e-7
    )
    assert deep_loss(bad, target)[0].item() > loss.item()


def test_coverage_supervision_recovers_from_opaque_saturation() -> None:
    head = DeepHead(8, nn.Identity(), layers=4)
    bias = head.output.bias
    assert bias is not None
    with torch.no_grad():
        bias.reshape(4, 5)[:, 1] = 30
    features = torch.zeros((1, 8, 8, 8))
    image = torch.zeros((1, 3, 8, 8))
    prediction = head(features, image, torch.ones((1, 1, 8, 8)))
    target = {
        "inverse_depth": prediction["inverse_depth"].detach(),
        "alpha": torch.full((1, 4, 8, 8), 0.2),
        "rgb": prediction["rgb"].detach(),
        "beauty": image,
        "mixed": torch.ones((1, 8, 8)),
    }
    _, terms = deep_loss(prediction, target)
    api.autograd.backward(terms["alpha"])
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad.reshape(4, 5)[:, 1], torch.full((4,), 0.2))
