from pathlib import Path
from typing import Any

import numpy as np
import torch

from strata_training.augment.deep import Shot, read_samples
from strata_training.augment.pipeline import AugmentConfig, Roles, recompose
from strata_training.augment.torch_compose import TensorShot, recompose_samples
from strata_training.augment.torch_deep import TensorSamples


def test_unlabelled_samples_keep_colour_geometry_and_roles_during_object_grading(
    packed_shot: Path,
    math_device: str,
) -> None:
    deep = read_samples(packed_shot / "deep/0001.exr")
    from strata_training.augment.deep import flatten

    image = flatten(deep)[0].astype(np.float32)
    shot = Shot(deep, image, image)
    config = AugmentConfig(
        clean=0,
        recompose=0,
        occluders=0,
        instance_grade=1,
        instance_count=(100, 100),
        scene_grade=0,
        grain=0,
        camera=0,
    )
    roles = Roles(frozenset({42}), ())
    record: dict[str, Any] = {}
    _, target = recompose(
        shot, roles, deep.targets(), None, np.random.default_rng(19), config, record
    )
    assert -1 not in record["graded_instances"]
    np.testing.assert_array_equal(target.rgb[0, :, 0, 0], deep.rgb[0])
    np.testing.assert_array_equal(target.alpha, deep.targets().alpha)
    np.testing.assert_array_equal(target.inverse_depth, deep.targets().inverse_depth)
    tensors = TensorSamples(
        deep.shape,
        *(
            torch.tensor(getattr(deep, name), device=math_device)
            for name in ("counts", "z", "alpha", "rgb", "ids", "roles")
        ),
    )
    tensor_image = torch.tensor(image, device=math_device)
    tensor_shot = TensorShot(tensors, tensor_image, tensor_image)
    record = {}
    _, rebuilt = recompose_samples(
        tensor_shot,
        roles,
        tensors.targets(),
        None,
        None,
        np.random.default_rng(19),
        config,
        record,
    )
    assert isinstance(rebuilt, TensorSamples)
    assert -1 not in record["graded_instances"]
    torch.testing.assert_close(
        rebuilt.rgb[tensors.ids == -1], tensors.rgb[tensors.ids == -1], atol=0, rtol=0
    )
    for name in ("z", "alpha", "roles", "ids"):
        torch.testing.assert_close(getattr(rebuilt, name), getattr(tensors, name), atol=0, rtol=0)
