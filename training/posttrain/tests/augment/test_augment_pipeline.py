from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from strata_training.augment.deep import Layer, Shot, compose, flatten, from_unsorted
from strata_training.augment.pipeline import (
    AugmentConfig,
    AugmentedSample,
    Donor,
    Roles,
    augment_sample,
    roles,
)
from strata_training.data.targets import composite
from strata_training.types import Array

OFF = AugmentConfig(
    clean=0.0,
    recompose=0.0,
    occluders=0.0,
    instance_grade=0.0,
    scene_grade=0.0,
    grain=0.0,
    camera=0.0,
)
ROLES = Roles(frozenset({2, 3}), ((1000, 50),))


def layered_shot(seed: int, size: tuple[int, int] = (24, 32)) -> Shot:
    rng = np.random.default_rng(seed)
    pixels = size[0] * size[1]
    rows = np.arange(pixels) // size[1]
    hero = (rows > size[0] // 3) & (np.arange(pixels) % size[1] < size[1] // 2)
    pixel = np.concatenate((np.arange(pixels), np.flatnonzero(hero), np.arange(pixels)))
    count = pixel.size
    alpha = np.concatenate(
        (rng.uniform(0.2, 0.6, pixels), np.ones(hero.sum()), np.ones(pixels))
    ).astype(np.float32)
    z = np.concatenate(
        (np.full(pixels, 3.0), np.full(hero.sum(), 6.0), np.full(pixels, 20.0))
    ).astype(np.float32)
    ids = np.concatenate((np.full(pixels, 1001), np.full(hero.sum(), 2), np.ones(pixels, int)))
    colour = rng.uniform(0.05, 0.9, (count, 3)).astype(np.float32) * alpha[:, None]
    deep = from_unsorted(size, pixel, z, alpha, colour, ids)
    image, _ = flatten(deep)
    return Shot(deep, image.astype(np.float32), image.astype(np.float32))


def test_empty_native_sky_keeps_image_without_inventing_occluder_depth() -> None:
    empty = np.empty(0, dtype=np.float32)
    deep = from_unsorted(
        (8, 8),
        empty.astype(np.int64),
        empty,
        empty,
        np.empty((0, 3), dtype=np.float32),
        empty.astype(np.int64),
    )
    image = np.full((3, 8, 8), 0.25, dtype=np.float32)
    shot = Shot(deep, image, image)
    sample = augment_sample(
        shot, ROLES, deep.targets(), None, np.random.default_rng(71), replace(OFF, occluders=1.0)
    )
    assert_array_equal(sample.image, image)
    assert not sample.layers.alpha.any()
    assert sample.record["occluders_skipped"] == "no_geometry"


def test_config_rejects_unknown_and_invalid_settings() -> None:
    assert AugmentConfig.from_dict({"grain_isos": [800, 1600]}).grain_isos == (800, 1600)
    with pytest.raises(ValueError, match="Unknown augmentation"):
        AugmentConfig.from_dict({"jpeg": 0.5})
    with pytest.raises(ValueError, match="studio group"):
        AugmentConfig.from_dict({"grain_isos": [123]})
    with pytest.raises(ValueError, match="codec_weights"):
        AugmentConfig.from_dict({"codec_weights": [1.0, 1.0]})


def test_roles_follow_manifest_instance_ranges() -> None:
    manifest = {
        "instance_ids": [
            {"name": "terrain", "first_id": 1, "count": 1},
            {"name": "hero_skin", "first_id": 2, "count": 1},
            {"name": "hero_groom", "first_id": 3, "count": 1},
            {"name": "grass", "first_id": 1000, "count": 10, "prototypes": ["a"]},
        ]
    }
    parsed = roles(manifest)
    assert parsed.hero == frozenset({2, 3})
    assert parsed.is_scatter(np.array([1, 3, 1000, 1009, 1010])).tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]


def test_photometric_augmentations_never_change_targets() -> None:
    shot = layered_shot(1)
    reference = shot.deep.targets()
    config = replace(OFF, grain=1.0, camera=1.0, sharpen=1.0)
    for seed in range(4):
        sample = augment_sample(
            shot, ROLES, shot.deep.targets(), None, np.random.default_rng(seed), config
        )
        for name, value in reference.arrays().items():
            assert_array_equal(sample.layers.arrays()[name], value)
        assert_allclose(sample.clean, shot.denoised, rtol=1e-6)
        assert np.abs(sample.image - shot.denoised).mean() > 1e-4


def test_scene_grade_changes_target_colour_with_the_input() -> None:
    shot = layered_shot(2)
    reference = shot.deep.targets()
    sample = augment_sample(
        shot,
        ROLES,
        shot.deep.targets(),
        None,
        np.random.default_rng(3),
        replace(OFF, scene_grade=1.0),
    )
    gains = np.array(sample.record["scene_gain"])
    assert_array_equal(sample.layers.alpha, reference.alpha)
    assert_array_equal(sample.layers.inverse_depth, reference.inverse_depth)
    assert_allclose(sample.layers.rgb, gains[None, :, None, None] * reference.rgb, rtol=1e-5)
    assert_allclose(sample.image, gains[:, None, None] * shot.denoised, rtol=1e-5)
    rebuilt = compose([Layer(shot.deep, shot, np.diag(gains))], shot, np.diag(gains))
    assert_allclose(sample.image, rebuilt.image, rtol=1e-5)
    for name, value in rebuilt.samples.targets().arrays().items():
        assert_allclose(sample.layers.arrays()[name], value, rtol=1e-5, atol=1e-7)


def test_recomposition_places_the_donor_hero_in_front_with_consistent_input() -> None:
    shot, donor = layered_shot(3), layered_shot(4)
    config = replace(OFF, recompose=1.0, hero=1.0, element_front=1.0)
    sample = augment_sample(
        shot, ROLES, shot.deep.targets(), Donor(donor, ROLES), np.random.default_rng(5), config
    )
    assert sample.record["element"] == "hero"
    assert sample.record["element_depth_scale"] < 3.0 / 6.0
    hero = np.zeros(donor.deep.shape, dtype=bool)
    hero.ravel()[np.unique(donor.deep.pixel[donor.deep.ids == 2])] = True
    alpha = sample.layers.alpha
    assert_allclose((1 - (1 - alpha[0]) * (1 - alpha[1]))[hero], 1, atol=1e-6)
    front_depth = 1 / sample.layers.inverse_depth[0][hero]
    assert np.all(front_depth < 3.0)
    composite = flatten_layers(sample)
    assert_allclose(sample.clean, composite, rtol=1e-4, atol=1e-5)


def flatten_layers(sample: AugmentedSample) -> Array:
    return composite(sample.layers.alpha, sample.layers.rgb)


def test_occluders_and_grades_rebuild_exact_targets_deterministically() -> None:
    shot, donor = layered_shot(6), layered_shot(7)
    config = replace(OFF, recompose=1.0, occluders=1.0, instance_grade=1.0, grain=1.0)
    first = augment_sample(
        shot, ROLES, shot.deep.targets(), Donor(donor, ROLES), np.random.default_rng(8), config
    )
    second = augment_sample(
        shot, ROLES, shot.deep.targets(), Donor(donor, ROLES), np.random.default_rng(8), config
    )
    assert first.record == second.record
    assert_array_equal(first.image, second.image)
    for name, value in first.layers.arrays().items():
        assert_array_equal(second.layers.arrays()[name], value)
    assert "occluders" in first.record
    assert_allclose(first.clean, flatten_layers(first), rtol=1e-4, atol=1e-5)


def test_clean_share_returns_the_untouched_sample() -> None:
    shot = layered_shot(9)
    sample = augment_sample(
        shot, ROLES, shot.deep.targets(), None, np.random.default_rng(0), AugmentConfig(clean=1.0)
    )
    assert sample.record == {"clean": True}
    assert_array_equal(sample.image, shot.denoised)
    assert_array_equal(sample.layers.alpha, shot.deep.targets().alpha)


def test_recomposition_rejects_a_donor_on_another_grid() -> None:
    config = replace(OFF, recompose=1.0)
    with pytest.raises(ValueError, match="pixel grid"):
        augment_sample(
            layered_shot(1),
            ROLES,
            layered_shot(1).deep.targets(),
            Donor(layered_shot(2, (24, 30)), ROLES),
            np.random.default_rng(0),
            config,
        )
