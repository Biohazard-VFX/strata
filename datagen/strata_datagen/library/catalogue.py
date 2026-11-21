"""The prepared asset library of a build root: what scenes may use, with provenance (assets/library.json)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

LIBRARY_FILE = "assets/library.json"


@dataclass(frozen=True)
class Provenance:
    source: str
    license: str
    origin: str


@dataclass(frozen=True)
class Model:
    key: str
    asset_id: str
    name: str
    role: str
    file: str
    size_m: tuple[float, float, float]
    triangles: int
    textures: dict[str, str]
    provenance: Provenance


@dataclass(frozen=True)
class Surface:
    asset_id: str
    name: str
    tile_m: float
    textures: dict[str, str]
    provenance: Provenance


@dataclass(frozen=True)
class Hdri:
    key: str
    file: str
    sun_direction: tuple[float, float, float]
    sun_irradiance: float
    sun_color: tuple[float, float, float]
    sky_irradiance: float
    provenance: Provenance


@dataclass(frozen=True)
class HeroPose:
    pose: str
    file: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    hairs: int


@dataclass(frozen=True)
class Hero:
    key: str
    albedo: str
    poses: tuple[HeroPose, ...]
    provenance: Provenance


@dataclass(frozen=True)
class Library:
    models: tuple[Model, ...]
    surfaces: tuple[Surface, ...]
    hdris: tuple[Hdri, ...]
    heroes: tuple[Hero, ...]

    def save(self, root: Path) -> None:
        path = root / LIBRARY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, sort_keys=True) + "\n")

    def role(self, role: str) -> tuple[Model, ...]:
        return tuple(m for m in self.models if m.role == role)

    def hero(self, key: str) -> Hero:
        return next(h for h in self.heroes if h.key == key)


Json = dict[str, object]


def _vec(raw: object) -> tuple[float, float, float]:
    x, y, z = cast("list[float]", raw)
    return (float(x), float(y), float(z))


def _provenance(raw: object) -> Provenance:
    return Provenance(**cast("dict[str, str]", raw))


def _model(raw: Json) -> Model:
    return Model(
        key=str(raw["key"]),
        asset_id=str(raw["asset_id"]),
        name=str(raw["name"]),
        role=str(raw["role"]),
        file=str(raw["file"]),
        size_m=_vec(raw["size_m"]),
        triangles=int(cast("int", raw["triangles"])),
        textures=cast("dict[str, str]", raw["textures"]),
        provenance=_provenance(raw["provenance"]),
    )


def _surface(raw: Json) -> Surface:
    return Surface(
        asset_id=str(raw["asset_id"]),
        name=str(raw["name"]),
        tile_m=float(cast("float", raw["tile_m"])),
        textures=cast("dict[str, str]", raw["textures"]),
        provenance=_provenance(raw["provenance"]),
    )


def _hdri(raw: Json) -> Hdri:
    return Hdri(
        key=str(raw["key"]),
        file=str(raw["file"]),
        sun_direction=_vec(raw["sun_direction"]),
        sun_irradiance=float(cast("float", raw["sun_irradiance"])),
        sun_color=_vec(raw["sun_color"]),
        sky_irradiance=float(cast("float", raw["sky_irradiance"])),
        provenance=_provenance(raw["provenance"]),
    )


def _hero(raw: Json) -> Hero:
    poses = tuple(
        HeroPose(
            pose=str(p["pose"]),
            file=str(p["file"]),
            bounds_min=_vec(p["bounds_min"]),
            bounds_max=_vec(p["bounds_max"]),
            hairs=int(cast("int", p["hairs"])),
        )
        for p in cast("list[Json]", raw["poses"])
    )
    return Hero(
        key=str(raw["key"]), albedo=str(raw["albedo"]), poses=poses, provenance=_provenance(raw["provenance"])
    )


def load(root: Path) -> Library:
    raw = cast("Json", json.loads((root / LIBRARY_FILE).read_text()))
    return Library(
        models=tuple(_model(m) for m in cast("list[Json]", raw["models"])),
        surfaces=tuple(_surface(s) for s in cast("list[Json]", raw["surfaces"])),
        hdris=tuple(_hdri(h) for h in cast("list[Json]", raw["hdris"])),
        heroes=tuple(_hero(h) for h in cast("list[Json]", raw["heroes"])),
    )
