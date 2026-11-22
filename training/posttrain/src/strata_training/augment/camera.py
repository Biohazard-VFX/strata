from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from strata_training.types import Array

av: Any = import_module("av")

AP1 = ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044))
REC709 = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
ARRI_WIDE_GAMUT_3 = ((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020))
D60 = (0.32168, 0.33767)
D65 = (0.3127, 0.3290)
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)
AP1_LUMINANCE = np.array([0.2722287168, 0.6740817658, 0.0536895174])
LOGC3 = {"cut": 0.010591, "a": 5.555556, "b": 0.052272, "c": 0.247190, "d": 0.385537}
LOGC3_LINEAR = (5.367655, 0.092809)
CURVES = ("rec709", "logc3")
CODECS = ("none", "jpeg", "h264", "h265")


def _xyz(xy: tuple[float, float]) -> Array:
    return np.array([xy[0] / xy[1], 1.0, (1 - xy[0] - xy[1]) / xy[1]])


def rgb_to_xyz(primaries: tuple[tuple[float, float], ...], white: tuple[float, float]) -> Array:
    columns = np.stack([_xyz(p) for p in primaries], axis=1)
    return columns * np.linalg.solve(columns, _xyz(white))


def conversion(
    source: tuple[tuple[float, float], ...],
    source_white: tuple[float, float],
    target: tuple[tuple[float, float], ...],
    target_white: tuple[float, float],
) -> Array:
    cone_source, cone_target = BRADFORD @ _xyz(source_white), BRADFORD @ _xyz(target_white)
    adapt = np.linalg.solve(BRADFORD, np.diag(cone_target / cone_source) @ BRADFORD)
    return np.linalg.solve(
        rgb_to_xyz(target, target_white), adapt @ rgb_to_xyz(source, source_white)
    )


AP1_TO_REC709 = conversion(AP1, D60, REC709, D65)
AP1_TO_AWG3 = conversion(AP1, D60, ARRI_WIDE_GAMUT_3, D65)


def transform(matrix: Array, image: Array) -> Array:
    return np.tensordot(matrix.astype(image.dtype), image, axes=1)


def logc3_encode(linear: Array) -> Array:
    cut, a, b, c, d = LOGC3.values()
    e, f = LOGC3_LINEAR
    safe = np.maximum(a * linear + b, 1e-10)
    return np.where(linear > cut, c * np.log10(safe) + d, e * linear + f)


def logc3_decode(code: Array) -> Array:
    cut, a, b, c, d = LOGC3.values()
    e, f = LOGC3_LINEAR
    return np.where(code > e * cut + f, (10 ** ((code - d) / c) - b) / a, (code - f) / e)


def tonemap(linear: Array, white: float) -> Array:
    return linear * (1 + linear / white**2) / (1 + linear)


def tonemap_inverse(display: Array, white: float) -> Array:
    k = 1 / white**2
    rest = 1 - display
    return (np.sqrt(rest * rest + 4 * k * display) - rest) / (2 * k)


def rec709_oetf(linear: Array) -> Array:
    clipped = np.maximum(linear, 0)
    return np.where(clipped < 0.018, 4.5 * clipped, 1.099 * np.power(clipped, 0.45) - 0.099)


def rec709_inverse_oetf(code: Array) -> Array:
    return np.where(code < 0.081, code / 4.5, ((code + 0.099) / 1.099) ** (1 / 0.45))


def encode(image: Array, curve: str, white: float) -> Array:
    if curve == "logc3":
        return logc3_encode(transform(AP1_TO_AWG3, image))
    display = transform(AP1_TO_REC709, image)
    return rec709_oetf(tonemap(np.clip(display, 0, white), white))


def decode(code: Array, curve: str, white: float) -> Array:
    if curve == "logc3":
        return transform(np.linalg.inv(AP1_TO_AWG3), logc3_decode(code))
    linear = tonemap_inverse(rec709_inverse_oetf(np.clip(code, 0, 1)), white)
    return transform(np.linalg.inv(AP1_TO_REC709), linear)


def tone(image: Array, contrast: float, saturation: float) -> Array:
    luminance = np.tensordot(AP1_LUMINANCE.astype(image.dtype), image, axes=1)
    graded = luminance + saturation * (image - luminance)
    positive = np.maximum(graded, 0)
    return np.where(graded > 0, 0.18 * (positive / 0.18) ** contrast, graded)


def unsharp(code: Array, amount: float, sigma: float) -> Array:
    return code + amount * (code - gaussian_filter(code, (0, sigma, sigma), mode="nearest"))


def quantize(code: Array, bits: int) -> Array:
    levels = (1 << bits) - 1
    return np.round(np.clip(code, 0, 1) * levels) / levels


def jpeg(code: Array, quality: int) -> Array:
    pixels = (quantize(code, 8) * 255).astype(np.uint8).transpose(1, 2, 0)
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=quality, subsampling=2)
    decoded: Array = np.asarray(Image.open(BytesIO(buffer.getvalue())).convert("RGB"))
    return decoded.transpose(2, 0, 1).astype(np.float32) / 255


def video(code: Array, codec: str, crf: int, bits: int) -> Array:
    height, width = code.shape[-2:]
    padded = np.pad(code, ((0, 0), (0, height % 2), (0, width % 2)), mode="edge")
    deep = bits > 8
    levels = 65535 if deep else 255
    rgb = np.round(np.clip(padded, 0, 1) * levels).astype(np.uint16 if deep else np.uint8)
    frame = av.VideoFrame.from_ndarray(
        np.ascontiguousarray(rgb.transpose(1, 2, 0)), format="rgb48le" if deep else "rgb24"
    )
    pix_fmt = "yuv420p10le" if deep else "yuv420p"
    encoder = av.CodecContext.create("libx265" if codec == "h265" else "libx264", "w")
    encoder.width, encoder.height = padded.shape[2], padded.shape[1]
    encoder.pix_fmt = pix_fmt
    encoder.thread_count = 1
    params = "keyint=1:log-level=error" if codec == "h265" else "keyint=1"
    encoder.options = {
        "crf": str(crf),
        "preset": "veryfast",
        ("x265-params" if codec == "h265" else "x264-params"): params,
    }
    packets = [*encoder.encode(frame.reformat(format=pix_fmt)), *encoder.encode(None)]
    decoder = av.CodecContext.create("hevc" if codec == "h265" else "h264", "r")
    decoder.thread_count = 1
    frames = [f for packet in [*packets, None] for f in decoder.decode(packet)]
    decoded = frames[0].to_ndarray(format="rgb48le" if deep else "rgb24")
    return decoded.transpose(2, 0, 1)[:, :height, :width].astype(np.float32) / levels


@dataclass(frozen=True)
class CameraParams:
    contrast: float
    saturation: float
    curve: str
    white: float
    sharpen_amount: float
    sharpen_sigma: float
    codec: str
    quality: int
    bits: int


def degrade(image: Array, params: CameraParams) -> Array:
    if params.curve not in CURVES or params.codec not in CODECS:
        raise ValueError("Unknown camera curve or codec")
    graded = tone(image.astype(np.float32), params.contrast, params.saturation)
    code = encode(graded, params.curve, params.white)
    if params.sharpen_amount > 0:
        code = unsharp(code, params.sharpen_amount, params.sharpen_sigma)
    if params.codec == "jpeg":
        code = jpeg(code, params.quality)
    elif params.codec == "none":
        code = quantize(code, params.bits)
    else:
        code = video(code, params.codec, params.quality, params.bits)
    return decode(code, params.curve, params.white).astype(np.float32)
