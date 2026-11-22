import io
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from strata_training.eval.real.protocol import validate_matte
from strata_training.types import Array


@dataclass(frozen=True)
class Sample:
    """Own decoded native arrays; no borrowed file handles; instances are independent."""

    identifier: str
    image: Array
    alpha: Array
    foreground: Array | None = None
    background: Array | None = None
    sequence: str | None = None
    frame: int | None = None


def decode_alpha(image: Image.Image) -> Array:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] == 4:
            array = array[..., 3]
        elif np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(
            array[..., 0], array[..., 2]
        ):
            array = array[..., 0]
        else:
            raise ValueError("Coloured alpha label is not grayscale")
    if array.dtype.kind not in "ui":
        raise ValueError("Expected integer alpha image")
    maximum = 65535 if image.mode.startswith("I;16") else 255
    return validate_matte(array.astype(np.float64) / maximum)


def natural_samples(root: Path) -> Iterator[Sample]:
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["eval_only"] or manifest["status"] != "downloaded":
        raise ValueError("Dataset is not a completed eval-only acquisition")
    images = {
        Path(f["path"]).stem: root / f["path"]
        for f in manifest["files"]
        if f["path"].startswith("image/")
    }
    alphas = {
        Path(f["path"]).stem: root / f["path"]
        for f in manifest["files"]
        if f["path"].startswith("alpha/")
    }
    if images.keys() != alphas.keys() or len(images) != manifest["expected"]:
        raise ValueError("Native RGB/alpha split pairing is incomplete")
    for key in sorted(images):
        with Image.open(images[key]) as file:
            image = np.asarray(file.convert("RGB"), dtype=np.float32) / 255
        with Image.open(alphas[key]) as file:
            alpha = decode_alpha(file)
        if image.shape[:2] != alpha.shape:
            raise ValueError("Native RGB/alpha resolution mismatch")
        yield Sample(key, image, alpha)


def video_samples(root: Path, frames_per_sequence: int | None = None) -> Iterator[Sample]:
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["eval_only"] or manifest["status"] != "downloaded":
        raise ValueError("Video is not a completed eval-only acquisition")
    with tarfile.open(root / "eval.tar") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        alpha_names = sorted(name for name in members if "/pha/" in name)
        if not alpha_names:
            raise ValueError("No video alpha frames found")

        def image_at(name: str, alpha: bool = False) -> Array:
            stream = archive.extractfile(members[name])
            if stream is None:
                raise ValueError("Missing tar image")
            with stream, Image.open(io.BytesIO(stream.read())) as image:
                return (
                    decode_alpha(image)
                    if alpha
                    else np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                )

        counts: dict[str, int] = {}
        for alpha_name in alpha_names:
            sequence = str(Path(alpha_name).parent)
            if frames_per_sequence is not None and counts.get(sequence, 0) >= frames_per_sequence:
                continue
            counts[sequence] = counts.get(sequence, 0) + 1
            candidates = [
                name
                for name in (
                    alpha_name.replace("/pha/", "/com/"),
                    alpha_name.replace("/pha/", "/com/").rsplit(".", 1)[0] + ".jpg",
                )
                if name in members
            ]
            if not candidates:
                raise ValueError("Video composite missing; never score cutout RGB as natural input")
            alpha = image_at(alpha_name, True)
            image = image_at(candidates[0])
            if image.shape[:2] != alpha.shape:
                raise ValueError("Video RGB/alpha resolution mismatch")
            path = Path(alpha_name)
            foreground_name = alpha_name.replace("/pha/", "/fgr/")
            if foreground_name not in members:
                foreground_name = foreground_name.rsplit(".", 1)[0] + ".jpg"
            foreground = image_at(foreground_name) if foreground_name in members else None
            background_name = alpha_name.replace("/pha/", "/bgr/")
            if "/videomatte_static/" in alpha_name:
                background_name = str(path.parent.parent / "bgr" / "0000.jpg")
            background = image_at(background_name) if background_name in members else None
            yield Sample(
                alpha_name,
                image,
                alpha,
                foreground=foreground,
                background=background,
                sequence=str(path.parent),
                frame=int(path.stem),
            )


def parquet_samples(root: Path) -> Iterator[Sample]:
    parquet: Any = import_module("pyarrow.parquet")
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["eval_only"] or manifest["status"] != "downloaded":
        raise ValueError("Parquet is not a completed eval-only acquisition")
    count = 0
    for item in manifest["files"]:
        file: Any = parquet.ParquetFile(root / item["path"])
        for batch in file.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            with Image.open(io.BytesIO(row["image"]["bytes"])) as image_file:
                image = np.asarray(image_file.convert("RGB"), dtype=np.float32) / 255
            with Image.open(io.BytesIO(row["mask"]["bytes"])) as alpha_file:
                alpha = decode_alpha(alpha_file)
            if image.shape[:2] != alpha.shape:
                raise ValueError("Parquet RGB/alpha resolution mismatch")
            yield Sample(str(count), image, alpha)
            count += 1
    if count != manifest["expected"]:
        raise ValueError("Parquet split row count mismatch")


@dataclass(frozen=True)
class LayeredSample:
    """Own a real RGB image and relative tuples; this dataset supplies no alpha GT."""

    identifier: str
    image: Array
    tuples: dict[str, Any]


def layered_samples(root: Path) -> Iterator[LayeredSample]:
    parquet: Any = import_module("pyarrow.parquet")
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["eval_only"] or manifest["status"] != "downloaded":
        raise ValueError("LayeredDepth is not a completed eval-only acquisition")
    count = 0
    for item in manifest["files"]:
        file: Any = parquet.ParquetFile(root / item["path"])
        for batch in file.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            with Image.open(io.BytesIO(row["image.png"]["bytes"])) as image_file:
                image = np.asarray(image_file.convert("RGB"), dtype=np.float32) / 255
            yield LayeredSample(row["__key__"], image, row["tuples.json"])
            count += 1
    if count != manifest["expected"]:
        raise ValueError("LayeredDepth validation row count mismatch")


def samples(root: Path, frames_per_sequence: int | None = None) -> Iterator[Sample]:
    if frames_per_sequence is not None and frames_per_sequence < 1:
        raise ValueError("frames_per_sequence must be positive")
    if root.name == "videomatte240k":
        return video_samples(root, frames_per_sequence)
    if root.name == "p3m500p":
        return parquet_samples(root)
    return natural_samples(root)
