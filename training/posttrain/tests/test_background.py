import numpy as np
import torch
from numpy.testing import assert_allclose

from strata_training.data.targets import reduce_identified_samples
from strata_training.losses.deep import deep_loss
from strata_training.torch_api import api


def test_sky_depth_is_masked_without_discarding_its_colour_or_coverage() -> None:
    layers = reduce_identified_samples(
        np.array([2.0, 3000.0]),
        np.array([0.25, 1.0]),
        np.array([[0.1, 0.05, 0.0], [0.6, 0.5, 0.4]]),
        np.array([2]),
        (1, 1),
        np.array([2, 0]),
        2,
    )
    assert_allclose(layers.inverse_depth[:, 0, 0], [0.5, 0])
    assert_allclose(layers.alpha[:, 0, 0], [0.25, 1])
    assert_allclose(layers.rgb[1, :, 0, 0], [0.6, 0.5, 0.4])
    depth = torch.tensor([0.5, 100.0]).reshape(1, 2, 1, 1).requires_grad_()
    logits = torch.zeros(1, 2, 1, 1, requires_grad=True)
    rgb = torch.zeros(1, 2, 3, 1, 1, requires_grad=True)
    prediction = {
        "inverse_depth": depth,
        "alpha_logits": logits,
        "alpha": logits.sigmoid(),
        "rgb": rgb,
    }
    target = {k: torch.tensor(v)[None] for k, v in layers.arrays().items()}
    target.update(beauty=torch.ones(1, 3, 1, 1), mixed=torch.ones(1, 1, 1))
    loss, terms = deep_loss(prediction, target)
    api.autograd.backward(loss)
    assert terms["depth"].item() == 0
    assert terms["gradient"].item() == 0
    assert depth.grad is not None
    assert depth.grad[0, 1, 0, 0].item() == 0
    assert logits.grad is not None
    assert logits.grad[0, 1, 0, 0].item() < 0
    assert rgb.grad is not None
    assert torch.all(rgb.grad[0, 1] < 0).item()


def test_geometry_and_sky_merged_into_one_cluster_mask_that_cluster_depth() -> None:
    layers = reduce_identified_samples(
        np.array([2.0, 3000.0]),
        np.array([0.25, 1.0]),
        np.ones((2, 3)),
        np.array([2]),
        (1, 1),
        np.array([1, 0]),
        1,
    )
    assert_allclose(layers.inverse_depth, 0)
    assert_allclose(layers.alpha, 1)
    assert_allclose(layers.rgb, 1.75)
