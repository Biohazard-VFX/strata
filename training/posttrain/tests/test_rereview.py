from pathlib import Path

import numpy as np
import pytest
import torch
from test_review_regressions import prepared_frames

from strata_training.data.online import OnlineLoader
from strata_training.data.targets import composite
from strata_training.data.torch_online import finalize, prepare_raw


@pytest.mark.parametrize("clean_fraction", [0.0, 1.0])
def test_every_view_uses_final_layer_composite(
    packed_shot: Path, tmp_path: Path, math_device: str, clean_fraction: float
) -> None:
    frames, samples = prepared_frames(packed_shot, tmp_path)
    from strata_training.data.dataset import Frame
    from strata_training.io import save_arrays

    for frame in frames:
        cache = Path(frame.entry["cache"]).with_name(frame.entry["id"] + ".rounded.npz")
        with np.load(frame.entry["cache"]) as archive:
            values = {k: archive[k] for k in archive.files}
        values["target_beauty"] = values["beauty"].astype(np.float32) + np.float32(2**-14)
        save_arrays(cache, **values)
        frame.entry = {**frame.entry, "cache": str(cache)}
    frames = [Frame(frame.entry, frame.feature_root) for frame in frames]
    loader = OnlineLoader(
        frames,
        samples,
        clean_fraction,
        dict.fromkeys(
            ("scene_grade", "instance_grade", "recompose", "occluders", "grain", "camera"), 0
        ),
    )
    selections = [(0, 0, 0), (1, 0, 0)]
    numpy = loader.prepare(selections, np.random.default_rng(71), 1, 0, "cpu")
    raw = prepare_raw(loader, selections, np.random.default_rng(71), 1, 0)
    raw.images = raw.images.to(math_device)
    raw.target = {k: v.to(math_device) for k, v in raw.target.items()}
    raw.jobs = [job.map(lambda value: value.to(math_device)) for job in raw.jobs]
    tensor = finalize(raw, 1, 0)
    from strata_training.data.dataset import batch as cached_batch

    _, cached, _ = cached_batch(frames, selections, 1, 0, "cpu")
    for targets in (numpy.target, tensor.target, cached):
        for position in range(2):
            expected = composite(
                targets["alpha"][position].cpu().numpy(),
                targets["rgb"][position].cpu().numpy(),
            )
            np.testing.assert_array_equal(targets["beauty"][position].cpu().numpy(), expected)
    for key, value in numpy.target.items():
        torch.testing.assert_close(value, tensor.target[key].cpu(), atol=0, rtol=0)


def test_removed_parameters_are_absent() -> None:
    import inspect

    from strata_training.augment.pipeline import augment_sample, recompose
    from strata_training.augment.torch_compose import recompose as tensor_recompose
    from strata_training.augment.torch_compose import recompose_samples
    from strata_training.data.scale import oracle_metric_scale

    for function in (
        augment_sample,
        recompose,
        tensor_recompose,
        recompose_samples,
    ):
        assert "shot_roles" not in inspect.signature(function).parameters
    assert "entry" not in inspect.signature(oracle_metric_scale).parameters


def test_loader_reports_eligibility_and_configurable_warning(
    packed_shot: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    frames, samples = prepared_frames(packed_shot, tmp_path)
    with caplog.at_level(logging.INFO):
        loader = OnlineLoader(frames, samples, eligibility_warning_fraction=0.75)
    assert loader.eligible == frozenset({0})
    assert "1/2" in caplog.text
    assert "50.0%" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        OnlineLoader(frames, samples, eligibility_warning_fraction=0.25)
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    for threshold in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="eligibility_warning_fraction"):
            OnlineLoader(frames, samples, eligibility_warning_fraction=threshold)


def test_hero_group_marker_is_named_once() -> None:
    import ast
    import inspect

    from strata_training.augment import pipeline, torch_compose

    markers: list[str] = []
    for module in (pipeline, torch_compose):
        tree = ast.parse(inspect.getsource(module))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and node.operand.value == 2
            ):
                parent = parents[node]
                if isinstance(parent, ast.Tuple):
                    continue
                assert isinstance(parent, ast.Assign)
                assert [target.id for target in parent.targets if isinstance(target, ast.Name)] == [
                    "HERO_GROUP"
                ]
                markers.append(module.__name__)
    assert markers == [pipeline.__name__]


def test_shared_composite_matches_independent_layer_weights(math_device: str) -> None:
    rng = np.random.default_rng(4891)
    alpha = rng.uniform(0, 1, (4, 8, 12)).astype(np.float32)
    rgb = rng.uniform(-1, 4, (4, 3, 8, 12)).astype(np.float32) * alpha[:, None]
    transmission = np.concatenate((np.ones_like(alpha[:1]), np.cumprod(1 - alpha[:-1], axis=0)))
    expected = (transmission[:, None] * rgb).sum(axis=0)
    np.testing.assert_array_equal(composite(alpha, rgb), expected)
    actual = composite(
        torch.tensor(alpha, device=math_device), torch.tensor(rgb, device=math_device)
    )
    assert isinstance(actual, torch.Tensor)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)
