import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from strata_dataset.samples import read

from strata_training.augment.deep import FIELDS, DeepSamples
from strata_training.data.cache import prepare
from strata_training.data.dataset import Frame
from strata_training.data.online import OnlineLoader, image_digest, prepare_sample
from strata_training.data.torch_online import finalize, prepare_raw
from strata_training.io import save_array, save_arrays, write_json


def prepared_frames(shot: Path, root: Path) -> tuple[list[Frame], Path]:
    entries = prepare(shot.parent, root / "targets")
    samples = root / "samples"
    features = root / "features"
    frames: list[Frame] = []
    for entry in entries:
        prepare_sample(entry, None, samples / entry["id"])
        image = np.load(samples / entry["id"] / "denoised.npy")
        target = features / entry["id"]
        save_arrays(
            target / "base.npz",
            raw_inverse_depth=np.ones(image.shape[1:], np.float32),
            raw_metric_scale=np.array(1.0),
        )
        save_array(target / "image.npy", image)
        save_array(target / "features.npy", np.zeros((1, *image.shape[1:]), np.float32))
        for level in range(4):
            save_array(target / f"pyramid_{level}.npy", np.zeros((1, 1, 1), np.float32))
        write_json(target / "context.json", {"image_sha256": image_digest(image)})
        frames.append(Frame(entry, features))
    return frames, samples


@pytest.mark.parametrize("mode", ["numpy", "torch"])
def test_stride_routing_keeps_off_stride_targets_and_seed_replay(
    packed_shot: Path, tmp_path: Path, mode: str
) -> None:
    frames, samples = prepared_frames(packed_shot, tmp_path)
    loader = OnlineLoader(
        frames,
        samples,
        0,
        {
            "scene_grade": 0,
            "grain": 0,
            "camera": 0,
            "instance_grade": 1,
            "recompose": 1,
            "occluders": 1,
        },
    )
    assert [f.entry["recomposition_eligible"] for f in frames] == [True, False]
    assert loader.eligible == (0,)
    assert loader.eligible_fraction == 0.5
    selections = [(0, 0, 0), (1, 0, 0)]

    def batch() -> Any:
        rng = np.random.default_rng(71)
        if mode == "torch":
            raw = prepare_raw(loader, selections, rng, 1, 0)
            off = raw.jobs[1]
            assert off.shot.deep is None
            assert off.donor is None
            assert not off.config.recompose
            assert not off.config.occluders
            assert not off.config.instance_grade
            raw.jobs = [job.map(lambda value: value) for job in raw.jobs]
            return finalize(raw, 1, 0)
        return loader.prepare(selections, rng, 1, 0, "cpu")

    actual, replay = batch(), batch()
    assert actual.live == [0, 1]
    for key in ("alpha", "rgb", "inverse_depth"):
        torch.testing.assert_close(
            actual.target[key][1],
            torch.tensor(frames[1].arrays[key][..., :1, :1]).float(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(actual.target[key], replay.target[key], atol=0, rtol=0)
    torch.testing.assert_close(actual.images, replay.images, atol=0, rtol=0)
    np.testing.assert_array_equal(actual.images[1].numpy(), loader.shot(1).denoised[:, :1, :1])
    torch.testing.assert_close(
        actual.target["beauty"][1],
        torch.tensor(frames[1].arrays["target_beauty"][:, :1, :1]).float(),
        atol=0,
        rtol=0,
    )
    from dataclasses import replace

    from strata_training.data.workers import NativeBatches, batches

    loader.clean_fraction = 0.4
    loader.augmentation = replace(loader.augmentation, occluders=0)
    config = {
        "seed": 719,
        "loader_math": mode,
        "crop": 1,
        "halo": 0,
        "mode": "head",
        "steps": 4,
        "batch_size": 4,
        "loader_workers": 0,
        "loader_prefetch": 2,
        "device": "cpu",
    }
    expected = [row.to("cpu") for row in batches(NativeBatches(frames, loader, [[0], [1]], config))]
    parallel = [
        row.to("cpu")
        for row in batches(
            NativeBatches(frames, loader, [[0], [1]], {**config, "loader_workers": 2})
        )
    ]
    resumed = [
        row.to("cpu") for row in batches(NativeBatches(frames, loader, [[0], [1]], config, 2))
    ]
    for a, b in [*zip(expected, parallel, strict=True), *zip(expected[2:], resumed, strict=True)]:
        assert a.selections == b.selections
        assert a.prepared is not None
        assert b.prepared is not None
        assert a.prepared.clean == b.prepared.clean
        torch.testing.assert_close(a.prepared.images, b.prepared.images, atol=0, rtol=0)
        for key, value in a.target.items():
            torch.testing.assert_close(value, b.target[key], atol=0, rtol=0)


def test_raw_exr_without_publication_is_never_recomposed(packed_shot: Path, tmp_path: Path) -> None:
    entry = prepare(packed_shot.parent, tmp_path / "targets")[0]
    sidecar = packed_shot / "samples" / entry["id"]
    sidecar.rename(sidecar.with_name("unpublished"))
    output = tmp_path / "samples"
    prepare_sample(entry, None, output)
    assert not (output / "deep").exists()


@pytest.mark.parametrize(
    "changed", ["manifest", "camera", "packed", "beauty", "samples", "version"]
)
def test_target_cache_rebuilds_for_changed_content(
    packed_shot: Path, tmp_path: Path, changed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strata_training.data import cache
    from strata_training.data.exr import exr

    entries = prepare(packed_shot.parent, tmp_path / "targets")
    path = Path(entries[0]["cache"])
    before = path.read_bytes()
    if changed in {"manifest", "camera"}:
        source = packed_shot / f"{changed}.json"
        record = json.loads(source.read_text())
        if changed == "manifest":
            record["color_space"] = "linear"
        else:
            record["frames"][0]["fx"] *= 2
        write_json(source, record)
    elif changed in {"packed", "beauty"}:
        source = packed_shot / ("layers" if changed == "packed" else "beauty") / "0001.exr"
        with exr.File(str(source), separate_channels=True) as file:
            file.parts[0].channels["L0.R" if changed == "packed" else "R"].pixels[:] *= 0.5
            file.write(str(source.with_suffix(".new.exr")))
        source.with_suffix(".new.exr").replace(source)
    elif changed == "samples":
        identity = packed_shot / "samples" / entries[0]["id"] / "identity.json"
        identity.write_text(identity.read_text() + "\n")
    else:
        monkeypatch.setattr(cache, "CACHE_VERSION", "test-next")
    updated = prepare(packed_shot.parent, tmp_path / "targets")
    rebuilt = Path(updated[0]["cache"])
    assert rebuilt != path
    assert rebuilt.read_bytes() != before
    assert path.read_bytes() == before
    path = rebuilt
    with np.load(path) as data:
        identity = str(data["content_identity"])
    prepare(packed_shot.parent, tmp_path / "targets")
    with np.load(path) as data:
        assert str(data["content_identity"]) == identity


def test_torch_reducer_matches_published_producer_samples(packed_shot: Path) -> None:
    from strata_training.augment.torch_deep import TensorSamples

    directory = next((packed_shot / "samples").iterdir())
    arrays = read(directory).arrays
    deep = DeepSamples(
        (int(arrays["shape"][0]), int(arrays["shape"][1])), **{k: arrays[k] for k in FIELDS}
    )
    tensor = TensorSamples(
        deep.shape,
        *(torch.tensor(getattr(deep, k)) for k in ("counts", "z", "alpha", "rgb", "ids", "roles")),
    )
    for key, expected in deep.targets().arrays().items():
        np.testing.assert_array_equal(tensor.targets()[key].numpy(), expected)


def test_torch_reducer_matches_real_render_crop() -> None:
    from strata_training.augment.torch_deep import TensorSamples

    record = json.loads((Path(__file__).parent / "fixtures/render_crop_040.json").read_text())
    deep = DeepSamples(
        (1, 1),
        *(
            np.array(record[k], dtype=np.int64 if k in {"offsets", "ids"} else np.float32)
            for k in FIELDS
        ),
    )
    tensor = TensorSamples(
        deep.shape,
        *(torch.tensor(getattr(deep, k)) for k in ("counts", "z", "alpha", "rgb", "ids", "roles")),
    )
    for key, expected in deep.targets().arrays().items():
        np.testing.assert_array_equal(tensor.targets()[key].numpy(), expected)


@pytest.mark.parametrize("mode", ["numpy", "torch"])
def test_unlabelled_hero_role_is_graded_without_grading_nonhero(mode: str) -> None:
    from strata_training.augment.deep import Shot, flatten
    from strata_training.augment.pipeline import AugmentConfig, recompose, roles
    from strata_training.augment.torch_compose import TensorShot, recompose_samples
    from strata_training.augment.torch_deep import TensorSamples

    deep = DeepSamples(
        (1, 2),
        np.array([0, 1, 2]),
        np.ones(2, np.float32),
        np.ones(2, np.float32),
        np.full((2, 3), 0.25, np.float32),
        np.full(2, -1, np.int64),
        np.array([[1, 1, 1], [1, 0, 0]], np.float32),
    )
    image = flatten(deep)[0]
    config = AugmentConfig(instance_grade=1, scene_grade=0, recompose=0, occluders=0)
    shot_roles = roles({"instance_ids": [{"name": "not_a_hero", "first_id": 42, "count": 1}]})
    if mode == "numpy":
        _, result = recompose(
            Shot(deep, image, image),
            shot_roles,
            deep.targets(),
            None,
            np.random.default_rng(42),
            config,
            {},
        )
        rgb = result.rgb
    else:
        tensor = TensorSamples(
            deep.shape,
            *(
                torch.tensor(getattr(deep, k))
                for k in ("counts", "z", "alpha", "rgb", "ids", "roles")
            ),
        )
        shot = TensorShot(tensor, torch.tensor(image), torch.tensor(image))
        _, rebuilt = recompose_samples(
            shot, shot_roles, tensor.targets(), None, None, np.random.default_rng(42), config, {}
        )
        assert isinstance(rebuilt, TensorSamples)
        rgb = rebuilt.targets()["rgb"].numpy()
    assert not np.array_equal(rgb[0, :, 0, 0], deep.rgb[0])
    np.testing.assert_array_equal(rgb[0, :, 0, 1], deep.rgb[1])


def test_torch_sky_uses_the_shared_role_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    from strata_dataset import roles

    from strata_training.augment.torch_deep import TensorSamples

    observed: list[bool] = []
    original = roles.sky

    def shared(alpha: Any, foreground: Any) -> Any:
        observed.append(True)
        return original(alpha, foreground)

    monkeypatch.setattr(roles, "sky", shared)
    samples = TensorSamples(
        (1, 1),
        torch.ones(1, dtype=torch.long),
        torch.ones(1),
        torch.ones(1),
        torch.ones((1, 3)),
        torch.zeros(1, dtype=torch.long),
        torch.zeros((1, 3)),
    )
    assert torch.equal(samples.sky, torch.ones(1, dtype=torch.bool))
    assert observed == [True]


@pytest.mark.parametrize("mode", ["numpy", "torch"])
def test_unlabelled_role_element_is_selected_without_manifest_names(mode: str) -> None:
    from strata_training.augment.deep import Shot, flatten
    from strata_training.augment.pipeline import AugmentConfig, Donor, recompose, roles
    from strata_training.augment.torch_compose import TensorShot, recompose_samples
    from strata_training.augment.torch_deep import TensorSamples

    deep = DeepSamples(
        (1, 1),
        np.array([0, 1]),
        np.ones(1, np.float32),
        np.ones(1, np.float32),
        np.full((1, 3), 0.25, np.float32),
        np.full(1, -1, np.int64),
        np.ones((1, 3), np.float32),
    )
    image = flatten(deep)[0]
    config = AugmentConfig(instance_grade=0, scene_grade=0, recompose=1, hero=1, occluders=0)
    shot_roles = roles({"instance_ids": []})
    record: dict[str, Any] = {}
    if mode == "numpy":
        shot = Shot(deep, image, image)
        recompose(
            shot,
            shot_roles,
            deep.targets(),
            Donor(shot, shot_roles),
            np.random.default_rng(42),
            config,
            record,
        )
    else:
        tensor = TensorSamples(
            deep.shape,
            *(
                torch.tensor(getattr(deep, k))
                for k in ("counts", "z", "alpha", "rgb", "ids", "roles")
            ),
        )
        tensor_shot = TensorShot(tensor, torch.tensor(image), torch.tensor(image))
        recompose_samples(
            tensor_shot,
            shot_roles,
            tensor.targets(),
            tensor_shot,
            shot_roles,
            np.random.default_rng(42),
            config,
            record,
        )
    assert record["element"] == "hero"


def test_published_samples_reject_changed_arrays(packed_shot: Path, tmp_path: Path) -> None:
    entry = prepare(packed_shot.parent, tmp_path / "targets")[0]
    source = packed_shot / "samples" / entry["id"] / "deep/z.npy"
    save_array(source, np.load(source) * 2)
    with pytest.raises(ValueError, match=r"identity\.json"):
        prepare_sample(entry, None, tmp_path / "samples")
