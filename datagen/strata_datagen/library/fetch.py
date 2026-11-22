"""Download Megascans assets that live on the Nextcloud library but are not in the local sync."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from strata_datagen.library.megascans import MAP_RE

REMOTE_ROOT = "/04_Library/Quixel/Downloaded"


def _nxc(*args: str) -> str:
    result = subprocess.run(["nxc", *args], capture_output=True, text=True, check=True, timeout=600)
    return result.stdout


def list_remote(remote: str) -> list[dict[str, object]]:
    listing = cast("dict[str, object]", json.loads(_nxc("files", "list", remote, "--format", "json")))
    return cast("list[dict[str, object]]", listing["entries"])


def _wanted_maps(names: Sequence[str], want_k: int) -> list[str]:
    best: dict[str, tuple[tuple[int, int, int, int], str]] = {}
    for name in names:
        match = MAP_RE.search(name)
        if match is None:
            continue
        res, channel, lod, ext = match.groups()
        k = int(res)
        rank = (
            0 if k >= want_k else 1,
            abs(k - want_k),
            0 if lod in (None, "0") else 1,
            int(ext.lower() == "exr"),
        )
        key = channel.lower()
        if key not in best or rank < best[key][0]:
            best[key] = (rank, name)
    return [entry[1] for entry in best.values()]


def _wanted_model(names: Sequence[str]) -> list[str]:
    lods = sorted(n for n in names if n.lower().endswith(".fbx") and "_LOD" in n)
    return lods[:1]


def wanted_files(names: Sequence[str], want_k: int = 2) -> list[str]:
    """Metadata, preview, one texture per channel and the most detailed LOD model."""
    keep = [n for n in names if n.endswith((".json", "_Preview.png"))]
    return sorted({*keep, *_wanted_maps(names, want_k), *_wanted_model(names)})


def fetch(category: str, folder: str, destination: Path) -> Path:
    """Mirror the needed files of one remote asset folder under destination/category/folder."""
    remote = f"{REMOTE_ROOT}/{category}/{folder}"
    target = destination / category / folder
    target.mkdir(parents=True, exist_ok=True)
    names = [str(e["name"]) for e in list_remote(remote) if not e.get("is_dir")]
    for name in wanted_files(names):
        local = target / name
        if not local.exists():
            _nxc("files", "download", f"{remote}/{name}", str(local), "--format", "json")
    return target
