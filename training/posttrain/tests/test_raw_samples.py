from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal


def test_native_raw_and_sidecar_samples_preserve_unlabelled_roles(
    tmp_path: Path,
    packed_shot: Path,
    math_device: str,
) -> None:
    import torch
    from strata_dataset.samples import read

    from strata_training.augment.deep import load_samples
    from strata_training.augment.deep import read_samples as read_native
    from strata_training.augment.torch_deep import TensorSamples
    from strata_training.data.compact import read_layers
    from strata_training.data.online import prepare_sample

    source = read(packed_shot / "samples/shot_0001").arrays
    deep = read_native(packed_shot / "samples/shot_0001")
    assert deep.ids[0] == -1
    assert_array_equal(deep.roles[0], [0.5, 0.25, 0.25])
    assert deep.ids[2] == 42
    assert_array_equal(deep.roles[2], [0.25, 0.25, 0.25])
    assert deep.ids[4] == 3
    assert_array_equal(deep.roles[4], [0.5, 0, 0])
    entry = {"id": "shot_0001", "frame": 1, "manifest": str(packed_shot / "manifest.json")}
    prepare_sample(entry, None, tmp_path / "samples")
    copied = load_samples(tmp_path / "samples/deep")
    for name in ("z", "alpha", "rgb", "ids", "roles", "offsets"):
        assert_array_equal(getattr(copied, name), source[name])
        assert_array_equal(getattr(deep, name), source[name])
    expected = read_layers(packed_shot / "layers/0001.exr")
    tensors = TensorSamples(
        deep.shape,
        *(
            torch.tensor(getattr(deep, name), device=math_device)
            for name in ("counts", "z", "alpha", "rgb", "ids", "roles")
        ),
    )
    from dataclasses import replace

    unlabelled = replace(deep, ids=np.full_like(deep.ids, -1))
    unlabelled_tensors = replace(tensors, ids=torch.full_like(tensors.ids, -1))
    for actual in (
        deep.targets().arrays(),
        tensors.targets(),
        unlabelled.targets().arrays(),
        unlabelled_tensors.targets(),
    ):
        for name, value in expected.arrays().items():
            a = actual[name]
            np.testing.assert_allclose(
                a.cpu().numpy() if isinstance(a, torch.Tensor) else a, value, atol=2e-6, rtol=2e-6
            )
    assert deep.targets().inverse_depth[0, 0, 0] == 0.5
