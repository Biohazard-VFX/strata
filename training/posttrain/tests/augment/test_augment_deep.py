from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from strata_training.augment.deep import (
    DeepSamples,
    Layer,
    Shot,
    compose,
    flatten,
    from_unsorted,
    load_samples,
    merge,
    read_samples,
    save_samples,
)
from strata_training.data.targets import reduce_samples

exr: Any = __import__("OpenEXR")


def random_deep(
    rng: np.random.Generator, shape: tuple[int, int], per_pixel: int, ids: tuple[int, ...]
) -> DeepSamples:
    pixels = shape[0] * shape[1]
    counts = rng.integers(0, per_pixel + 1, pixels)
    total = int(counts.sum())
    alpha = rng.uniform(0.05, 1.0, total).astype(np.float32)
    colour = rng.uniform(0, 1, (total, 3)).astype(np.float32) * alpha[:, None]
    return from_unsorted(
        shape,
        np.repeat(np.arange(pixels), counts),
        rng.uniform(1, 50, total).astype(np.float32),
        alpha,
        colour,
        rng.choice(np.array(ids), total),
    )


def test_flatten_of_merge_matches_independent_front_to_back_composite() -> None:
    rng = np.random.default_rng(3)
    a, b = random_deep(rng, (5, 7), 4, (2,)), random_deep(rng, (5, 7), 4, (1,))
    merged, labels = merge([a, b])
    rgb, alpha = flatten(merged)
    order = np.argsort(np.concatenate((a.pixel, b.pixel)), kind="stable")
    reference = reduce_samples(
        np.concatenate((a.z, b.z))[order].astype(np.float64),
        np.concatenate((a.alpha, b.alpha))[order].astype(np.float64),
        np.concatenate((a.rgb, b.rgb))[order].astype(np.float64),
        a.counts + b.counts,
        (5, 7),
        layers=1,
    )
    assert_allclose(rgb, reference.rgb[0], atol=1e-6)
    assert_allclose(alpha, reference.alpha[0], atol=1e-6)
    assert np.bincount(labels).tolist() == [a.z.size, b.z.size]
    assert np.all(np.diff(merged.z)[np.diff(merged.pixel) == 0] >= 0)


def test_element_in_front_composites_over_background_algebraically() -> None:
    rng = np.random.default_rng(4)
    front = random_deep(rng, (4, 4), 3, (2,)).scaled(0.01)
    back = random_deep(rng, (4, 4), 3, (1,))
    merged, _ = merge([front, back])
    rgb, alpha = flatten(merged)
    front_rgb, front_alpha = flatten(front)
    back_rgb, back_alpha = flatten(back)
    assert_allclose(rgb, front_rgb + (1 - front_alpha) * back_rgb, atol=1e-6)
    assert_allclose(alpha, front_alpha + (1 - front_alpha) * back_alpha, atol=1e-6)


def test_crop_selection_and_storage_preserve_samples(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    deep = random_deep(rng, (6, 9), 5, (1, 2, 1000))
    window = deep.crop(1, 2, 4, 5)
    assert_allclose(flatten(window)[0], flatten(deep)[0][:, 1:5, 2:7])
    assert_allclose(window.targets().alpha, deep.targets().alpha[:, 1:5, 2:7])
    hero = deep.select(deep.ids == 2)
    assert set(hero.ids.tolist()) == {2}
    assert hero.z.size == int((deep.ids == 2).sum())
    save_samples(tmp_path / "deep", deep)
    loaded = load_samples(tmp_path / "deep")
    assert loaded.shape == deep.shape
    for name in ("offsets", "z", "alpha", "rgb", "ids"):
        assert_array_equal(getattr(loaded, name), getattr(deep, name))


def test_read_samples_decodes_premultiplied_instance_ids_and_sorts_by_depth(
    tmp_path: Path,
) -> None:
    channels = {n: np.empty((1, 2), dtype=object) for n in ("R", "G", "B", "A", "Z", "ZBack")}
    channels["instance_id"] = np.empty((1, 2), dtype=object)
    values: dict[str, tuple[list[float], list[float]]] = {
        "Z": ([5.0, 2.0], []),
        "A": ([0.5, 0.25], []),
        "instance_id": ([0.5 * 1000, 0.25 * 3], []),
    }
    for name, array in channels.items():
        for x in range(2):
            data = values.get(name, ([0.1, 0.2], []))[x]
            array[0, x] = np.array(data, dtype=np.float32)
    path = tmp_path / "deep.exr"
    exr.File({"type": exr.deepscanline, "compression": exr.ZIPS_COMPRESSION}, channels).write(
        str(path)
    )
    deep = read_samples(path)
    assert deep.counts.tolist() == [2, 0]
    assert_allclose(deep.z, [2.0, 5.0])
    assert deep.ids.tolist() == [3, 1000]


def shot_from(deep: DeepSamples, rng: np.random.Generator, noise: float) -> Shot:
    rgb, _ = flatten(deep)
    noisy = rgb + rng.normal(0, noise, rgb.shape) * rgb
    return Shot(deep, noisy.astype(np.float32), rgb.astype(np.float32))


def test_compose_reproduces_the_denoised_beauty_when_nothing_is_inserted() -> None:
    rng = np.random.default_rng(6)
    deep = random_deep(rng, (5, 6), 4, (1, 1000))
    shot = shot_from(deep, rng, 0.2)
    split = [
        Layer(deep.select(deep.ids == 1), shot),
        Layer(deep.select(deep.ids == 1000), shot),
    ]
    result = compose(split, shot, np.eye(3))
    assert_allclose(result.image, shot.denoised, rtol=1e-5, atol=1e-6)
    assert_allclose(flatten(result.samples)[0], flatten(deep)[0], atol=1e-6)


def test_compose_inserts_an_opaque_element_and_grades_its_samples() -> None:
    rng = np.random.default_rng(7)
    background = shot_from(random_deep(rng, (4, 5), 3, (1,)), rng, 0.1)
    pixels = np.arange(20)
    element_deep = from_unsorted(
        (4, 5),
        pixels,
        np.full(20, 0.5, dtype=np.float32),
        np.ones(20, dtype=np.float32),
        np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (20, 1)),
        np.full(20, 2),
    )
    element = shot_from(element_deep, rng, 0.0)
    grade = np.diag([2.0, 1.0, 0.5])
    result = compose(
        [Layer(element_deep, element, grade), Layer(background.deep, background)],
        background,
        np.eye(3),
    )
    expected = np.broadcast_to(np.array([0.4, 0.4, 0.3])[:, None, None], (3, 4, 5))
    assert_allclose(result.image, expected, rtol=1e-5)
    assert_allclose(flatten(result.samples)[0], expected, rtol=1e-5)


def test_compose_keeps_environment_visible_through_partial_inserts() -> None:
    empty = from_unsorted(
        (1, 1),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros(0, dtype=np.int64),
    )
    sky = np.full((3, 1, 1), 0.8, dtype=np.float32)
    environment = Shot(empty, sky, sky)
    glass = from_unsorted(
        (1, 1),
        np.zeros(1, dtype=np.int64),
        np.ones(1, dtype=np.float32),
        np.array([0.25], dtype=np.float32),
        np.array([[0.25, 0.0, 0.0]], dtype=np.float32),
        np.full(1, -1),
    )
    result = compose([Layer(glass, None), Layer(empty, environment)], environment, np.eye(3))
    assert_allclose(result.image[:, 0, 0], [0.25 + 0.75 * 0.8, 0.6, 0.6], rtol=1e-6)


def test_merge_rejects_mismatched_grids() -> None:
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError, match="identical pixel grids"):
        merge([random_deep(rng, (2, 2), 1, (1,)), random_deep(rng, (2, 3), 1, (1,))])
