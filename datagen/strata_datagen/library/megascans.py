"""Read-only catalogue of local Megascans downloads: metadata, texture maps and model files."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Category = Literal["3d", "3dplant", "surface"]
CHANNELS = ("Albedo", "Roughness", "Normal", "Opacity")
MAP_RE = re.compile(
    r"_(\d+)K_(" + "|".join(CHANNELS) + r")(?:_LOD(\d+))?\.(jpg|jpeg|png|exr)$", re.IGNORECASE
)
LOD_RE = re.compile(r"_LOD(\d+)\.fbx$", re.IGNORECASE)
SOURCE = "Quixel Megascans"
LICENSE = "Megascans free license; AI training use confirmed by owner"


@dataclass(frozen=True)
class TextureMap:
    channel: str
    path: Path
    resolution_k: int


@dataclass(frozen=True)
class Model:
    variation: int
    lod: int
    triangles: int
    path: Path


@dataclass(frozen=True)
class Asset:
    asset_id: str
    name: str
    category: Category
    folder: Path
    tags: tuple[str, ...]
    max_size_m: float | None
    maps: Mapping[str, TextureMap]
    models: tuple[Model, ...]


def _map_rank(found: TextureMap, ext: str, lod: int | None, want_k: int) -> tuple[int, int, int, int]:
    above = found.resolution_k >= want_k
    distance = found.resolution_k - want_k if above else -found.resolution_k
    return (0 if above else 1, distance, 0 if lod in (None, 0) else 1, int(ext.lower() == "exr"))


def find_maps(folder: Path, want_k: int = 2) -> dict[str, TextureMap]:
    """Pick one file per channel: the smallest resolution >= want_k, LOD0 or shared, JPEG over EXR."""
    best: dict[str, tuple[tuple[int, int, int, int], TextureMap]] = {}
    for parent in (folder, folder / "Textures" / "Atlas"):
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            match = MAP_RE.search(path.name)
            if match is None or path.stat().st_size == 0:
                continue
            res, raw_channel, lod, ext = match.groups()
            channel = next(c for c in CHANNELS if c.lower() == raw_channel.lower())
            found = TextureMap(channel, path, int(res))
            rank = _map_rank(found, ext, None if lod is None else int(lod), want_k)
            if channel not in best or rank < best[channel][0]:
                best[channel] = (rank, found)
    return {channel: entry[1] for channel, entry in best.items()}


def _model_path(folder: Path, uri: str) -> Path | None:
    for candidate in (folder / uri, folder / Path(uri).name):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def find_models(folder: Path, metadata: Mapping[str, object], max_triangles: int) -> tuple[Model, ...]:
    """Per variation, the most detailed downloaded FBX LOD within the triangle budget."""
    chosen: dict[int, Model] = {}
    for raw in cast("list[dict[str, object]]", metadata.get("models", [])):
        if raw.get("mimeType") != "application/x-fbx" or raw.get("type", "lod") != "lod":
            continue
        triangles = int(cast("int", raw.get("tris") or 0))
        path = _model_path(folder, str(raw["uri"]))
        if path is None or triangles <= 0 or triangles > max_triangles:
            continue
        model = Model(
            int(cast("int", raw.get("variation", 1))), int(cast("int", raw["lod"])), triangles, path
        )
        current = chosen.get(model.variation)
        if current is None or model.lod < current.lod:
            chosen[model.variation] = model
    return tuple(chosen[v] for v in sorted(chosen)) or _loose_model(folder)


def _loose_model(folder: Path) -> tuple[Model, ...]:
    """Single-variation downloads whose metadata lists no FBX: the most detailed LOD file found."""
    found = sorted((int(m[1]), p) for p in folder.glob("*.fbx") if (m := LOD_RE.search(p.name)))
    return (Model(1, found[0][0], 0, found[0][1]),) if found else ()


def _metadata_file(folder: Path) -> Path | None:
    files = sorted(p for p in folder.glob("*.json") if p.stat().st_size > 0)
    return files[0] if files else None


def _semantic(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return cast("Mapping[str, object]", metadata.get("semanticTags", {}))


def load_asset(folder: Path, category: Category, max_triangles: int = 40000) -> Asset | None:
    """Return the asset in folder, or None when its metadata or required files are missing."""
    meta_path = _metadata_file(folder)
    if meta_path is None:
        return None
    metadata = cast("dict[str, object]", json.loads(meta_path.read_text()))
    semantic = _semantic(metadata)
    maps = find_maps(folder)
    models = () if category == "surface" else find_models(folder, metadata, max_triangles)
    if "Albedo" not in maps or (category != "surface" and not models):
        return None
    size = semantic.get("maxSize")
    return Asset(
        asset_id=str(metadata["id"]),
        name=str(semantic.get("name") or metadata.get("name") or folder.name),
        category=category,
        folder=folder,
        tags=tuple(str(t) for t in cast("list[object]", metadata.get("tags", []))),
        max_size_m=float(cast("float", size)) if isinstance(size, int | float) and size else None,
        maps=maps,
        models=models,
    )


def scan(root: Path, category: Category) -> Iterator[Asset]:
    """Every usable asset of one category under a Megascans Downloaded root."""
    base = root / category
    if not base.is_dir():
        return
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        asset = load_asset(folder, category, 120000 if category == "3d" else 40000)
        if asset is not None:
            yield asset
