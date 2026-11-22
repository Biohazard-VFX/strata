import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.ndimage import gaussian_filter

from strata_training.augment.camera import (
    AP1_TO_REC709,
    CameraParams,
    decode,
    degrade,
    encode,
    jpeg,
    logc3_encode,
    quantize,
    tone,
    unsharp,
    video,
)


def smooth_plate(shape: tuple[int, int] = (64, 96), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.random((3, *shape)), (0, 4, 4))
    return (0.05 + 0.6 * (noise - noise.min()) / np.ptp(noise)).astype(np.float32)


def test_ap1_to_rec709_matches_aces_bradford_matrix() -> None:
    expected = [
        [1.7050509, -0.6217921, -0.0832588],
        [-0.1302564, 1.1408047, -0.0105483],
        [-0.0240033, -0.1289690, 1.1529723],
    ]
    assert_allclose(AP1_TO_REC709, expected, atol=2e-4)
    assert_allclose(AP1_TO_REC709.sum(axis=1), 1, atol=1e-6)


def test_logc3_matches_nuke_alexa_v3_logc() -> None:
    linear = np.array([-0.5, -0.17041015625, -0.0001220703125, 0.04931640625, 0.5986328125])
    nuke = [-2.591018, -0.821894, 0.092154, 0.265292, 0.516217]
    assert_allclose(logc3_encode(linear), nuke, atol=2e-6)


@pytest.mark.parametrize("curve", ["rec709", "logc3"])
def test_camera_curves_round_trip_in_range(curve: str) -> None:
    image = np.tensordot(np.linalg.inv(AP1_TO_REC709), smooth_plate(), axes=1)
    assert_allclose(decode(encode(image, curve, 8.0), curve, 8.0), image, rtol=1e-4, atol=1e-6)


def test_rec709_curve_clips_highlights_above_white() -> None:
    bright = np.full((3, 2, 2), 50.0, dtype=np.float32)
    back = decode(quantize(encode(bright, "rec709", 8.0), 8), "rec709", 8.0)
    assert np.all(back <= 8.0 * 1.001)


def test_tone_keeps_grey_and_neutral_axis() -> None:
    grey = np.full((3, 1, 1), 0.18, dtype=np.float32)
    assert_allclose(tone(grey, 1.2, 0.7), grey, rtol=1e-5)
    bright = np.full((3, 1, 1), 0.72, dtype=np.float32)
    assert_allclose(tone(bright, 1.2, 1.0), 0.18 * 4.0**1.2, rtol=1e-5)


def test_quantize_bands_to_code_values() -> None:
    ramp = np.linspace(0, 1, 5000, dtype=np.float32)[None, None, :].repeat(3, axis=0)
    assert np.unique(quantize(ramp, 8)).size == 256
    assert np.unique(quantize(ramp, 10)).size == 1024


def test_jpeg_uses_420_chroma_subsampling() -> None:
    columns = np.arange(64) % 2
    chroma = np.where(columns, 0.8, 0.2)[None, None, :] * np.array([1.0, 0.0, 1.0])[:, None, None]
    chroma = (
        chroma + np.where(columns, 0.0, 0.6)[None, None, :] * np.array([0, 1.0, 0])[:, None, None]
    )
    chroma = np.broadcast_to(chroma, (3, 32, 64)).astype(np.float32)
    luma = np.broadcast_to(np.where(columns, 0.8, 0.2), (3, 32, 64)).astype(np.float32)
    chroma_error = np.abs(jpeg(chroma, 95) - chroma).mean()
    luma_error = np.abs(jpeg(luma, 95) - luma).mean()
    assert chroma_error > 5 * luma_error


def test_unsharp_adds_overshoot_at_edges() -> None:
    edge = np.zeros((3, 8, 16), dtype=np.float32)
    edge[:, :, 8:] = 0.5
    sharpened = unsharp(edge, 0.5, 1.0)
    assert sharpened.max() > 0.5
    assert sharpened.min() < 0


@pytest.mark.parametrize(("codec", "bits"), [("h264", 8), ("h265", 8), ("h265", 10)])
def test_video_intra_codec_is_deterministic_and_rate_controlled(codec: str, bits: int) -> None:
    code = encode(smooth_plate((63, 95), 1), "rec709", 8.0)
    fine = video(code, codec, 12, bits)
    assert fine.shape == code.shape
    assert_allclose(fine, video(code, codec, 12, bits))
    coarse = np.abs(video(code, codec, 45, bits) - code).mean()
    assert np.abs(fine - code).mean() < coarse


def test_clean_degrade_is_near_identity_and_codecs_add_error() -> None:
    image = smooth_plate()
    clean = CameraParams(1.0, 1.0, "logc3", 8.0, 0.0, 1.0, "none", 0, 10)
    assert np.abs(degrade(image, clean) - image).max() < 0.005
    harsh = CameraParams(1.0, 1.0, "rec709", 8.0, 0.0, 1.0, "jpeg", 20, 8)
    assert (
        np.abs(degrade(image, harsh) - image).mean() > np.abs(degrade(image, clean) - image).mean()
    )
    with pytest.raises(ValueError, match="Unknown camera"):
        degrade(image, CameraParams(1.0, 1.0, "srgb", 8.0, 0.0, 1.0, "none", 0, 8))
