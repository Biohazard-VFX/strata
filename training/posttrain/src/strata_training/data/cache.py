import fcntl
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from functools import wraps
from os import stat_result
from pathlib import Path
from typing import Any, Concatenate

import numpy as np
import torch
from strata_dataset.samples import recorded_identity

from strata_training.data.compact import read_layers
from strata_training.data.encoder_input import display
from strata_training.data.exr import read_rgb, read_roles
from strata_training.data.metadata import camera_frames, category
from strata_training.data.policy import is_test_only
from strata_training.io import save_arrays, write_json
from strata_training.types import Array, Layers, fp16_safe

CACHE_VERSION = "targets-fable-float-v2"


@contextmanager
def cache_lock(output: Path, *, shared: bool) -> Generator[None]:
    """Hold a shared training lease or exclusive offline writer lease; never wait on users."""
    writable = os.access(output, os.W_OK)
    if not shared and not writable:
        raise PermissionError("Target cache maintenance requires a writable directory")
    with (output / ".cache.lock").open("a+b" if writable else "rb") as lock:
        try:
            fcntl.flock(lock, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            message = (
                "Target cache is in use by a writer or prune; finish maintenance before reading"
                if shared
                else "Target cache is in use; stop training before offline maintenance"
            )
            raise ValueError(message) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def cache_readers(*indexes: Path) -> Generator[None]:
    directories = {
        Path(entry["cache"]).resolve().parent
        for index in indexes
        for entry in json.loads(index.read_text())
    }
    with ExitStack() as leases:
        for directory in sorted(directories):
            leases.enter_context(cache_lock(directory, shared=True))
        for index in indexes:
            for entry in json.loads(index.read_text()):
                validate_target(Path(entry["cache"]))
        yield


def cache_reader[**P, R](
    function: Callable[Concatenate[Path, P], R],
) -> Callable[Concatenate[Path, P], R]:
    @wraps(function)
    def locked(index: Path, *args: P.args, **kwargs: P.kwargs) -> R:
        with cache_readers(index):
            return function(index, *args, **kwargs)

    return locked


def validate_target(path: Path) -> None:
    with np.load(path) as data:
        if "version" not in data or str(data["version"]) != CACHE_VERSION:
            raise ValueError("Target cache version mismatch; rebuild target caches")


def digest_stat(stat: stat_result) -> list[int]:
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns, stat.st_dev]


class Digests:
    def __init__(self, output: Path) -> None:
        self.path = output / "digests.json"
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.records: dict[str, Any] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )

    def digest(self, path: Path) -> str:
        stat = path.stat()
        key = digest_stat(stat)
        name = str(path.resolve())
        record = self.records.get(name)
        if (
            record is not None
            and record.get("boot_id") == self.boot_id
            and record["stat"] == key
            and all(
                abs(timestamp - record["hashed_ns"]) > 2_000_000_000
                for timestamp in (stat.st_mtime_ns, stat.st_ctime_ns)
            )
        ):
            return str(record["sha256"])
        hashed_ns = time.time_ns()
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        after = path.stat()
        if digest_stat(after) != key:
            raise ValueError("Source changed while hashing target cache input")
        self.records[name] = {
            "stat": key,
            "sha256": digest,
            "hashed_ns": hashed_ns,
            "boot_id": self.boot_id,
        }
        return digest

    def save(self) -> None:
        write_json(self.path, self.records)


def prune(output: Path) -> list[Path]:
    """Exclusively remove unreferenced target archives and mappings, never source data."""
    with cache_lock(output, shared=False):
        entries = json.loads((output / "index.json").read_text())
        referenced = {Path(entry["cache"]).resolve() for entry in entries}
        if any(path.parent != output.resolve() for path in referenced):
            raise ValueError("Target index references caches outside its directory")
        removed: list[Path] = []
        for path in sorted(output.iterdir()):
            archive = path.with_suffix("") if path.name.endswith(".npz.mmap") else path
            if archive.suffix != ".npz" or archive.resolve() in referenced:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
        return removed


def content_identity(
    shot: Path,
    manifest: dict[str, Any],
    camera: dict[str, Any],
    frame: int,
    layers: int,
    memo: Digests,
) -> str:
    hashes: dict[str, str] = {}
    for name in ("layers", "beauty"):
        hashes[name] = memo.digest(shot / name / f"{frame:04d}.exr")
    record = {
        "version": CACHE_VERSION,
        "layers": layers,
        "manifest": manifest,
        "camera": camera,
        "samples": recorded_identity(shot / "samples" / f"{shot.name}_{frame:04d}"),
        "files": hashes,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def prepare(dataset: Path, output: Path, layers: int = 4) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    with cache_lock(output, shared=False):
        return _prepare(dataset, output, layers)


def _prepare(dataset: Path, output: Path, layers: int) -> list[dict[str, Any]]:
    memo = Digests(output)
    entries: list[dict[str, Any]] = []
    manifests = sorted(dataset.glob("*/manifest.json"))
    ranked = sorted(manifests, key=lambda p: hashlib.sha256(p.parent.name.encode()).digest())
    heldout: set[str] = (
        {p.parent.name for p in ranked[: max(1, len(ranked) // 5)]} if len(ranked) > 1 else set()
    )
    for manifest_path in manifests:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        shot = manifest_path.parent
        if manifest["shot_id"] != shot.name:
            raise ValueError("Shot directory does not match its literal shot_id")
        protected = is_test_only(shot)
        split = (
            "test"
            if protected
            else manifest.get("split", "test" if shot.name in heldout else "train")
        )
        if split not in {"train", "test"}:
            raise ValueError("Shot split must be train or test")
        for frame, camera in sorted(camera_frames(shot).items()):
            identifier = f"{shot.name}_{int(frame):04d}"
            identity = content_identity(shot, manifest, camera, int(frame), layers, memo)
            cache = output / f"{identifier}.{identity}.npz"
            reusable = False
            if cache.exists():
                with np.load(cache) as existing:
                    reusable = (
                        "version" in existing
                        and str(existing["version"]) == CACHE_VERSION
                        and "content_identity" in existing
                        and str(existing["content_identity"]) == identity
                    )
            if not reusable:
                packed = shot / "layers" / f"{int(frame):04d}.exr"
                target = read_layers(packed, layers)
                image = read_rgb(shot / "beauty" / f"{int(frame):04d}.exr")
                expected = (camera["height"], camera["width"])
                if image.shape[-2:] != expected or target.alpha.shape[-2:] != expected:
                    raise ValueError(
                        f"Native beauty/deep/camera dimensions disagree in {identifier}"
                    )
                srgb = display_rgb(image, manifest.get("color_space", "ACEScg"))
                mixed = (target.alpha[0] > 0.03) & (target.alpha[0] < 0.97)
                reference = read_roles(packed)
                reference["denoised"] = read_rgb(
                    shot / "beauty" / f"{int(frame):04d}.exr", part_name="denoised"
                ).astype(np.float32)
                save_arrays(
                    cache,
                    version=np.array(CACHE_VERSION),
                    inverse_depth=target.inverse_depth,
                    alpha=target.alpha,
                    rgb=fp16_safe(target.rgb),
                    beauty=fp16_safe(image),
                    srgb=srgb,
                    mixed=mixed,
                    content_identity=np.array(identity),
                    **reference,
                )
            entries.append(
                {
                    "id": identifier,
                    "shot": shot.name,
                    "frame": int(frame),
                    "split": split,
                    "test_only": protected,
                    "dataset_root": str(dataset.resolve()),
                    "category": manifest.get("category", category(camera)),
                    "noise_varies_per_frame": camera["noise_varies_per_frame"],
                    "cache": str(cache.resolve()),
                    "manifest": str(manifest_path.resolve()),
                    "recomposition_eligible": recorded_identity(shot / "samples" / identifier)
                    is not None,
                    "compact_scene_only": False,
                }
            )
    if not entries:
        raise ValueError(f"No shot manifests found in {dataset}")
    memo.save()
    write_json(output / "index.json", entries)
    return entries


def display_rgb(image: Array, color_space: str) -> Array:
    return display(torch.as_tensor(image), color_space).numpy()


def load_layers(path: Path) -> tuple[Layers, dict[str, Array]]:
    with np.load(path) as file:
        arrays = {k: file[k] for k in file.files}
    return Layers(
        arrays["inverse_depth"], arrays["alpha"], arrays["rgb"].astype(np.float32)
    ), arrays
