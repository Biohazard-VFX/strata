from math import log

import pytest
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
    assert torch.all((result["alpha"] > 0) & (result["alpha"] <= 1))
    torch.testing.assert_close(result["alpha"][0, :, 0, 0], torch.tensor([0.25, 1 / 3, 0.5, 1]))


def test_supervision_masks_invalid_depth_but_penalizes_extra_coverage() -> None:
    alpha = torch.tensor([[[[0.5]], [[1.0]], [[0.0]], [[0.0]]]])
    inverse = torch.tensor([[[[0.5]], [[0.25]], [[0.0]], [[0.0]]]])
    rgb = alpha[:, :, None].expand(-1, -1, 3, -1, -1) * 0.5
    prediction = {
        "inverse_depth": inverse.clamp_min(0.01).requires_grad_(),
        "alpha": alpha.clone().requires_grad_(),
        "alpha_logits": torch.logit(alpha, eps=1e-7).requires_grad_(),
        "rgb": rgb.clone().requires_grad_(),
        "layer_rgb": rgb.clone().requires_grad_(),
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
    torch.testing.assert_close(loss, torch.tensor(0.9241963))
    api.autograd.backward(loss)
    assert prediction["alpha"].grad is not None
    bad = dict(prediction)
    bad["alpha_logits"] = torch.logit(
        alpha + torch.tensor([[[[0.0]], [[0.0]], [[0.5]], [[0.0]]]]), eps=1e-7
    )
    assert deep_loss(bad, target)[0].item() > loss.item()


@pytest.mark.parametrize("layers", [1, 2, 4])
def test_coverage_bce_averages_only_learned_slots(layers: int) -> None:
    alpha = torch.full((1, layers, 2, 2), 0.3)
    alpha[:, -1] = 1
    prediction = {
        "inverse_depth": torch.ones_like(alpha),
        "alpha": torch.full_like(alpha, 0.8),
        "alpha_logits": torch.full_like(alpha, log(4)),
        "rgb": torch.zeros((1, layers, 3, 2, 2)),
        "layer_rgb": torch.zeros((1, layers, 3, 2, 2)),
    }
    target = {
        "inverse_depth": torch.ones_like(alpha),
        "alpha": alpha,
        "rgb": prediction["rgb"],
        "beauty": torch.zeros((1, 3, 2, 2)),
        "mixed": torch.tensor([[[0, 1], [1, 0]]]),
    }
    _, terms = deep_loss(prediction, target)
    expected = 0 if layers == 1 else -0.3 * log(0.8) - 0.7 * log(0.2)
    assert terms["alpha"].item() == pytest.approx(expected)


def test_coverage_supervision_recovers_from_opaque_saturation() -> None:
    head = DeepHead(8, nn.Identity(), layers=4)
    bias = head.output.bias
    assert bias is not None
    with torch.no_grad():
        bias.reshape(4, 5)[:, 1] = torch.tensor([90, 60, 30, 0])
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
    assert torch.isfinite(bias.grad).all()
    assert bias.grad.reshape(4, 5)[0, 1] > 0.2
    assert bias.grad.reshape(4, 5)[-1, 1] < -0.2


@pytest.mark.parametrize("layers", range(1, 9))
@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_head_coverage_is_exactly_one(layers: int, scale: float) -> None:
    api.manual_seed(901)
    generator = torch.Generator().manual_seed(901)
    head = DeepHead(2, nn.Identity(), layers)
    with torch.no_grad():
        head.output.weight.copy_(torch.randn(head.output.weight.shape, generator=generator) * scale)
        assert head.output.bias is not None
        head.output.bias.copy_(torch.randn(layers * 5, generator=generator) * scale)
    features = torch.randn((2, 2, 8, 8), generator=generator)
    image = torch.ones((2, 3, 8, 8))
    result = head(features, image, torch.ones((2, 1, 8, 8)))
    alpha = result["alpha"]
    assert torch.isfinite(alpha).all()
    assert torch.all((alpha >= 0) & (alpha <= 1))
    assert torch.equal(1 - (1 - alpha).prod(dim=1), torch.ones((2, 8, 8)))
    assert torch.equal(alpha[:, -1], torch.ones_like(alpha[:, -1]))
    through = torch.cat((torch.ones_like(alpha[:, :1]), (1 - alpha[:, :-1]).cumprod(1)), 1)
    hidden = alpha[:, :-1][through[:, :-1] == 0]
    assert torch.equal(hidden, torch.zeros_like(hidden))
    torch.testing.assert_close((through * alpha).sum(1), torch.ones((2, 8, 8)), atol=2e-7, rtol=0)
    target = {
        "inverse_depth": result["inverse_depth"].detach(),
        "alpha": alpha.detach(),
        "rgb": result["rgb"].detach(),
        "mixed": torch.zeros((2, 8, 8)),
        "beauty": image,
    }
    loss, _ = deep_loss(result, target)
    api.autograd.backward(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    with torch.no_grad():
        head.output.weight.zero_()
        assert head.output.bias is not None
        logits = head.output.bias.reshape(layers, 5)[:, 1]
        expected = logits.softmax(0)
    regular = head(features, image, torch.ones((2, 1, 8, 8)))["alpha"]
    trans = torch.cat((torch.ones_like(regular[:, :1]), (1 - regular[:, :-1]).cumprod(1)), 1)
    torch.testing.assert_close((trans * regular)[0, :, 0, 0], expected, atol=2e-6, rtol=0)


def test_layer_colour_does_not_steer_coverage_but_composite_still_does() -> None:
    head = DeepHead(2, nn.Identity(), 4)
    image = torch.ones((1, 3, 8, 8))
    prediction = head(torch.zeros((1, 2, 8, 8)), image, torch.ones((1, 1, 8, 8)))
    target = {
        "inverse_depth": prediction["inverse_depth"].detach(),
        "alpha": prediction["alpha"].detach(),
        "rgb": torch.zeros_like(prediction["rgb"]),
        "beauty": torch.zeros_like(image),
        "mixed": torch.ones((1, 8, 8)),
    }
    _, terms = deep_loss(prediction, target)
    bias = head.output.bias
    assert bias is not None
    gradient = torch.autograd.grad(terms["rgb"], bias, retain_graph=True)[0].reshape(4, 5)
    assert torch.equal(gradient[:, 1], torch.zeros(4))
    assert (gradient[:, 2:] > 0).all()
    with torch.no_grad():
        bias.reshape(4, 5)[0, 2:] = 1
    prediction = head(torch.zeros((1, 2, 8, 8)), image, torch.ones((1, 1, 8, 8)))
    _, terms = deep_loss(prediction, target)
    gradient = torch.autograd.grad(terms["composite"], bias)[0].reshape(4, 5)
    torch.testing.assert_close(gradient[:, 1], torch.tensor([3, -1, -1, -1]) / 16)
