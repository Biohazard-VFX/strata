"""Human hero library: MPFB bodies with CC0 skins and clothes, animated with retargeted CMU clips.

A recipe (JSON, outside the repo) names the asset pools; every human is sampled from
(seed, index), built in Blender with MPFB (blender/mpfb_human.py), retargeted and skinned here,
and written as a hero directory: hero.json, rest.usdc, clips/<clip>.usdc and textures/.

Recipe fields:
  seed, count, key_prefix; blender, blender_user_resources; mpfb_data (extracted asset packs),
  mpfb_targets (MPFB target.json); cmu (folder of <subject>.asf and <trial>.amc);
  skins: [{id, sex, ages, ancestry}] (ages from teen, young, middle, old; ancestry african, asian, caucasian);
  outfits: {sex: {full, tops, bottoms, shoes: [mhclo id]}}, a full garment or a top and bottom, plus shoes;
  eyes: [eye mhmat id]; clip_frames;
  motions: {category: [{trial, start}]} (start is the first 120 fps source frame);
  licenses: [{source, license}] copied into every hero.json, one line per source;
  garmentcode (optional): GarmentRecipe fields; humans in its index range wear only shoes from the
  MPFB pools plus one GarmentCode design per sex, simulated in Vellum for every clip.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from strata_datagen import files
from strata_datagen.generators import materials
from strata_datagen.library import garments, human_regions, human_usd, mocap, retarget, textures
from strata_datagen.library.megascans import TextureMap
from strata_datagen.rng import stream
from strata_datagen.usdutil import FloatArray, IntArray

BLENDER_SCRIPT = Path(__file__).resolve().parents[2] / "blender" / "mpfb_human.py"
RIG = "cmu_mb"
FPS = 24.0
SUBFRAMES = np.array([-0.25, 0.0, 0.25])
HANDLES = 1
CATEGORIES = ("walk", "run", "gesture", "dance", "turn")
AGES = {"teen": (0.3, 0.42), "young": (0.45, 0.56), "middle": (0.62, 0.78), "old": (0.86, 1.0)}
ANCESTRIES = ("african", "asian", "caucasian")
SEXES = ("female", "male")
FACE_GROUPS = ("nose", "mouth", "chin", "eyes", "head", "cheek", "ears", "eyebrows", "forehead")
FACE_TARGETS = 10
FULL_OUTFIT_SHARE = 0.4
BLENDER_Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
SKIN_ROUGHNESS = 0.42
SKIN_SUBSURFACE = 0.35
SKIN_SUBSURFACE_RADIUS = (1.0, 0.37, 0.18)
SKIN_SUBSURFACE_SCALE_M = 0.006


@dataclass(frozen=True)
class Skin:
    id: str
    sex: str
    ages: tuple[str, ...]
    ancestry: str


@dataclass(frozen=True)
class MotionOption:
    trial: str
    start: int


@dataclass(frozen=True)
class Recipe:
    seed: int
    count: int
    key_prefix: str
    blender: str
    blender_user_resources: Path
    mpfb_data: Path
    mpfb_targets: Path
    cmu: Path
    skins: tuple[Skin, ...]
    outfits: dict[str, dict[str, tuple[str, ...]]]
    eyes: tuple[str, ...]
    clip_frames: int
    motions: dict[str, tuple[MotionOption, ...]]
    licenses: tuple[dict[str, str], ...]
    garmentcode: garments.GarmentRecipe | None = None

    def indices(self) -> tuple[int, ...]:
        extra = (
            ()
            if self.garmentcode is None
            else tuple(
                range(self.garmentcode.first_index, self.garmentcode.first_index + self.garmentcode.count)
            )
        )
        return tuple(range(self.count)) + extra

    @staticmethod
    def load(path: Path) -> "Recipe":
        raw = cast("dict[str, object]", json.loads(path.read_text()))
        skins = cast("list[dict[str, object]]", raw["skins"])
        outfits = cast("dict[str, dict[str, list[str]]]", raw["outfits"])
        motions = cast("dict[str, list[dict[str, object]]]", raw["motions"])
        return Recipe(
            seed=int(cast("int", raw["seed"])),
            count=int(cast("int", raw["count"])),
            key_prefix=str(raw["key_prefix"]),
            blender=str(raw["blender"]),
            blender_user_resources=Path(str(raw["blender_user_resources"])),
            mpfb_data=Path(str(raw["mpfb_data"])),
            mpfb_targets=Path(str(raw["mpfb_targets"])),
            cmu=Path(str(raw["cmu"])),
            skins=tuple(
                Skin(str(s["id"]), str(s["sex"]), tuple(cast("list[str]", s["ages"])), str(s["ancestry"]))
                for s in skins
            ),
            outfits={sex: {k: tuple(v) for k, v in pools.items()} for sex, pools in outfits.items()},
            eyes=tuple(cast("list[str]", raw["eyes"])),
            clip_frames=int(cast("int", raw["clip_frames"])),
            motions={
                category: tuple(MotionOption(str(m["trial"]), int(cast("int", m["start"]))) for m in options)
                for category, options in motions.items()
            },
            licenses=tuple(cast("list[dict[str, str]]", raw["licenses"])),
            garmentcode=None
            if "garmentcode" not in raw
            else garments.GarmentRecipe.parse(cast("dict[str, object]", raw["garmentcode"])),
        )


@dataclass(frozen=True)
class HumanSpec:
    key: str
    skin: Skin
    phenotype: dict[str, object]
    targets: tuple[tuple[str, float], ...]
    eyes: str
    outfit: tuple[str, ...]
    motions: dict[str, MotionOption]
    garments: tuple[Path, ...] = ()


def _pick[T](rng: np.random.Generator, options: tuple[T, ...]) -> T:
    return options[int(rng.integers(0, len(options)))]


Catalog = dict[str, dict[str, list[dict[str, object]]]]


def face_targets(catalog: Catalog, rng: np.random.Generator) -> list[tuple[str, float]]:
    """Signed face modifiers drawn from MPFB's target catalogue, sided ones applied symmetrically."""
    modifiers = [m for group in FACE_GROUPS for m in catalog[group]["categories"]]
    chosen = rng.choice(len(modifiers), FACE_TARGETS, replace=False)
    out: list[tuple[str, float]] = []
    for index in sorted(int(i) for i in chosen):
        modifier = modifiers[index]
        value = float(rng.uniform(-0.7, 0.7))
        if "opposites" not in modifier:
            out.append((cast("list[str]", modifier["targets"])[0], round(abs(value), 3)))
            continue
        opposites = cast("dict[str, str]", modifier["opposites"])
        sign = "positive" if value > 0.0 else "negative"
        sides = ("left", "right") if modifier["has_left_and_right"] else ("unsided",)
        out.extend((opposites[f"{sign}-{side}"], round(abs(value), 3)) for side in sides)
    return out


def _skin(skins: tuple[Skin, ...], index: int, rng: np.random.Generator) -> tuple[Skin, str]:
    """Skins stratified over ancestry, sex and age so every run of 24 humans covers each cell once."""
    ancestry = ANCESTRIES[index % len(ANCESTRIES)]
    sex = SEXES[index // len(ANCESTRIES) % len(SEXES)]
    age = tuple(AGES)[index // (len(ANCESTRIES) * len(SEXES)) % len(AGES)]
    matching = tuple(s for s in skins if s.ancestry == ancestry and s.sex == sex and age in s.ages)
    return _pick(rng, matching), age


def _outfit(pools: dict[str, tuple[str, ...]], rng: np.random.Generator) -> tuple[str, ...]:
    full = bool(rng.random() < FULL_OUTFIT_SHARE)
    top, bottom, garment = _pick(rng, pools["tops"]), _pick(rng, pools["bottoms"]), _pick(rng, pools["full"])
    return (garment, _pick(rng, pools["shoes"])) if full else (top, bottom, _pick(rng, pools["shoes"]))


def sample(recipe: Recipe, index: int, catalog: Catalog) -> HumanSpec:
    rng = stream(recipe.seed, "human", str(index))
    skin, age = _skin(recipe.skins, index, rng)
    male = skin.sex == "male"
    race: dict[str, float] = dict.fromkeys(ANCESTRIES, 0.1)
    race[skin.ancestry] = 0.8
    low, high = AGES[age]
    phenotype: dict[str, object] = {
        "gender": float(rng.uniform(0.8, 1.0) if male else rng.uniform(0.0, 0.2)),
        "age": float(rng.uniform(low, high)),
        "muscle": float(rng.uniform(0.25, 0.85)),
        "weight": float(rng.uniform(0.2, 0.85)),
        "proportions": float(rng.uniform(0.3, 0.8)),
        "height": float(rng.uniform(0.2, 0.85)),
        "cupsize": float(rng.uniform(0.3, 0.8)),
        "firmness": float(rng.uniform(0.3, 0.8)),
        "race": race,
    }
    targets = tuple(face_targets(catalog, rng))
    eyes = _pick(rng, recipe.eyes)
    outfit = _outfit(recipe.outfits[skin.sex], rng)
    motions = {c: _pick(rng, recipe.motions[c]) for c in CATEGORIES}
    sewn: tuple[Path, ...] = ()
    if recipe.garmentcode is not None and recipe.garmentcode.applies(index):
        sewn = (_pick(rng, recipe.garmentcode.designs[skin.sex]),)
        outfit = outfit[-1:]
    key = f"{recipe.key_prefix}{index:04d}"
    return HumanSpec(key, skin, phenotype, targets, eyes, outfit, motions, sewn)


def _asset(recipe: Recipe, kind: str, name: str, suffix: str) -> Path:
    return recipe.mpfb_data / kind / name / f"{name}{suffix}"


def _blender_job(recipe: Recipe, spec: HumanSpec, output: Path) -> dict[str, object]:
    return {
        "recipe": {
            "phenotype": spec.phenotype,
            "targets": [{"target": t, "value": v} for t, v in spec.targets],
            "rig": RIG,
            "skin": str(_asset(recipe, "skins", spec.skin.id, ".mhmat")),
            "eyes": str(recipe.mpfb_data / "eyes" / "high-poly" / "high-poly.mhclo"),
            "clothes": [str(_asset(recipe, "clothes", c, ".mhclo")) for c in spec.outfit],
        },
        "output": str(output),
    }


def run_blender(recipe: Recipe, jobs: list[dict[str, object]], scratch: Path) -> None:
    listing = scratch / "mpfb_jobs.json"
    files.write_text(listing, json.dumps(jobs))
    subprocess.run(
        [
            recipe.blender,
            "-b",
            "--python-exit-code",
            "1",
            "--python",
            str(BLENDER_SCRIPT),
            "--",
            str(listing),
        ],
        check=True,
        capture_output=True,
        timeout=7200,
        env={"BLENDER_USER_RESOURCES": str(recipe.blender_user_resources), "PATH": "/usr/bin:/bin"},
    )


@dataclass(frozen=True)
class RestPart:
    name: str
    kind: str
    source: str
    points: FloatArray
    counts: IntArray
    indices: IntArray
    uv: FloatArray
    joints: IntArray
    weights: FloatArray


@dataclass(frozen=True)
class RestHuman:
    rig: retarget.Rig
    parts: tuple[RestPart, ...]
    groups: dict[str, human_regions.Mask]


def _y_up(points: FloatArray) -> FloatArray:
    return points @ BLENDER_Z_UP_TO_Y_UP.T


def load_rest(path: Path) -> RestHuman:
    with np.load(path) as data:
        meta = cast("dict[str, object]", json.loads(str(data["meta"])))
        rig = retarget.Rig(
            tuple(cast("list[str]", meta["bones"])),
            data["bone_parents"].astype(np.int64),
            _y_up(data["bone_heads"]),
            _y_up(data["bone_tails"]),
        )
        parts = tuple(
            RestPart(
                str(p["name"]),
                str(p["kind"]),
                str(p["source"]),
                _y_up(data[f"{p['name']}/points"]),
                data[f"{p['name']}/counts"].astype(np.int64),
                data[f"{p['name']}/indices"].astype(np.int64),
                data[f"{p['name']}/uv"].astype(np.float64),
                data[f"{p['name']}/joints"].astype(np.int64),
                data[f"{p['name']}/weights"].astype(np.float64),
            )
            for p in cast("list[dict[str, str]]", meta["parts"])
        )
        groups = {k.removeprefix("body/region_"): data[k].astype(bool) for k in data.files if "/region_" in k}
    return RestHuman(rig, parts, groups)


def body_regions(human: RestHuman) -> dict[str, IntArray]:
    body = next(p for p in human.parts if p.kind == "body")
    eyes = next(p for p in human.parts if p.kind == "eyes").points
    names = human.rig.names

    def weight(bone: str) -> FloatArray:
        return np.where(body.joints == names.index(bone), body.weights, 0.0).sum(axis=1)

    marks = human_regions.Landmarks(eyes[eyes[:, 0] > 0].mean(axis=0), eyes[eyes[:, 0] < 0].mean(axis=0))
    masks = human_regions.regions(body.points, weight("Head"), weight("Neck1"), human.groups, marks)
    return {k: human_regions.face_subset(m, body.counts, body.indices) for k, m in masks.items()}


def _mhmat_textures(mhmat: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for line in mhmat.read_text(errors="replace").splitlines():
        words = line.split(maxsplit=1)
        if len(words) == 2 and words[0] in ("diffuseTexture", "normalmapTexture"):
            found[words[0]] = mhmat.parent / words[1].strip()
    return found


def _material_file(part: RestPart, eyes: str, recipe: Recipe) -> Path:
    if part.kind == "body":
        return Path(part.source)
    if part.kind == "eyes":
        return recipe.mpfb_data / "eyes" / "materials" / f"{eyes}.mhmat"
    source = Path(part.source)
    line = next(w for w in source.read_text(errors="replace").splitlines() if w.startswith("material "))
    return source.parent / line.split(maxsplit=1)[1].strip()


def _look(kind: str, folder: Path, mhmat: Path, relative_to: Path) -> materials.Look:
    maps = _mhmat_textures(mhmat)
    albedo = textures.convert(TextureMap("Albedo", maps["diffuseTexture"], 0), folder)
    normal = None
    if "normalmapTexture" in maps and maps["normalmapTexture"].exists():
        normal = textures.convert(TextureMap("Normal", maps["normalmapTexture"], 0), folder)
    found = materials.Textures(
        albedo.relative_to(relative_to, walk_up=True).as_posix(),
        normal=None if normal is None else normal.relative_to(relative_to, walk_up=True).as_posix(),
    )
    if kind == "body":
        return materials.Look(
            found,
            roughness=SKIN_ROUGHNESS,
            subsurface=SKIN_SUBSURFACE,
            subsurface_radius=SKIN_SUBSURFACE_RADIUS,
            subsurface_scale_m=SKIN_SUBSURFACE_SCALE_M,
        )
    if kind == "eyes":
        return materials.Look(found, roughness=0.08, specular=1.0)
    return materials.Look(found, roughness=0.8, specular=0.35)


@dataclass(frozen=True)
class Clip:
    """Skinned samples of every part at clip time codes; frame 0 and the last frame are handles."""

    name: str
    category: str
    option: MotionOption
    times: FloatArray
    samples: dict[str, FloatArray]
    track: FloatArray
    bounds_min: FloatArray
    bounds_max: FloatArray

    @property
    def frames(self) -> int:
        return len(self.track) - 2 * HANDLES


def clip_window(motion: mocap.Motion, option: MotionOption, frames: int) -> tuple[int, int]:
    """First source frame and clip length in frames (handles included) that fit inside the take."""
    step = mocap.FPS / FPS
    start = max(option.start, int(np.ceil(-SUBFRAMES[0] * step)))
    available = int(np.floor((len(motion.rotations) - 1 - start) / step - SUBFRAMES[-1])) + 1
    return start, min(frames + 2 * HANDLES, available)


def build_clip(human: RestHuman, category: str, option: MotionOption, frames: int, cmu: Path) -> Clip:
    skeleton, motion = mocap.load(cmu / f"{option.trial.split('_')[0]}.asf", cmu / f"{option.trial}.amc")
    start, length = clip_window(motion, option, frames)
    if length < 2 * HANDLES + 24:
        raise ValueError(f"{option.trial} from {option.start} is shorter than 24 frames")
    times = (np.arange(length, dtype=np.float64)[:, None] + SUBFRAMES[None, :]).ravel()
    pose = retarget.retarget(human.rig, skeleton, motion, start + times * mocap.FPS / FPS)
    samples = {p.name: retarget.skin(p.points, p.joints, p.weights, human.rig, pose) for p in human.parts}
    on_frame = np.flatnonzero(np.isclose(times, np.round(times)))
    root = pose.heads[on_frame, 0]
    lowest = min(float(s[:, :, 1].min()) for s in samples.values())
    centre = root[length // 2]
    offset = np.array([-centre[0], -lowest, -centre[2]])
    samples = {name: s + offset for name, s in samples.items()}
    track = root + offset
    local = [s[on_frame] - (track * np.array([1.0, 0.0, 1.0]))[:, None] for s in samples.values()]
    low = np.min([s.min(axis=(0, 1)) for s in local], axis=0)
    high = np.max([s.max(axis=(0, 1)) for s in local], axis=0)
    return Clip(f"{category}_{option.trial}", category, option, times, samples, track, low, high)


def _vec(values: FloatArray) -> list[float]:
    return [round(float(v), 5) for v in values]


def _clip_record(clip: Clip, file: str) -> dict[str, object]:
    duration = clip.frames / FPS
    travel = clip.track[-1 - HANDLES] - clip.track[HANDLES]
    return {
        "file": file,
        "category": clip.category,
        "source": f"CMU mocap trial {clip.option.trial} from source frame {clip.option.start}",
        "frames": clip.frames,
        "first_frame": HANDLES,
        "fps": FPS,
        "subframes": _vec(SUBFRAMES),
        "speed_mps": round(float(np.hypot(travel[0], travel[2]) / duration), 4),
        "bounds_min": _vec(clip.bounds_min),
        "bounds_max": _vec(clip.bounds_max),
        "track": [_vec(p) for p in clip.track],
    }


def _parts(human: RestHuman) -> list[human_usd.MeshPart]:
    return [human_usd.MeshPart(p.name, p.counts, p.indices, p.uv, p.points, p.name) for p in human.parts]


@dataclass(frozen=True)
class Sewn:
    """A GarmentCode garment: flat panels with seams, its mesh part and its fabric look."""

    recipe: garments.GarmentRecipe
    design: str
    panels: Path
    part: human_usd.MeshPart
    look: materials.Look


def _sew(recipe: Recipe, spec: HumanSpec, human: RestHuman, folder: Path) -> list[Sewn]:
    if not spec.garments or recipe.garmentcode is None:
        return []
    gc = recipe.garmentcode
    body = next(p for p in human.parts if p.kind == "body")
    sex = spec.skin.sex
    template = cast("dict[str, float]", json.loads(gc.templates[sex].read_text())["body"])
    heads = {name: human.rig.heads[i] for i, name in enumerate(human.rig.names)}
    build = folder / "build"
    garments.write_measurements(build / "body.yaml", garments.measure(body.points, heads, template))
    jobs: list[dict[str, object]] = [
        {
            "body": str(build / "body.yaml"),
            "design": str(design),
            "name": design.stem,
            "output": str(build / f"panels_{design.stem}.npz"),
            "resolution_cm": gc.resolution_cm,
        }
        for design in spec.garments
    ]
    garments.run_panels(gc, jobs, build)
    out: list[Sewn] = []
    for index, design in enumerate(spec.garments):
        panels = build / f"panels_{design.stem}.npz"
        with np.load(panels) as data:
            triangles = data["triangles"].astype(np.int64)
            uv, points = data["uv"].astype(np.float64), data["points"].astype(np.float64)
        name = f"sewn{index}"
        texture = garments.fabric_texture(
            folder / "textures" / name / "albedo.jpg", stream(recipe.seed, "fabric", spec.key, design.stem)
        )
        part = human_usd.MeshPart(
            name, np.full(len(triangles), 3), triangles.ravel(), uv[triangles.ravel()], points, name
        )
        look = materials.Look(
            materials.Textures(texture.relative_to(folder / "usd", walk_up=True).as_posix()),
            roughness=0.85,
            specular=0.3,
        )
        out.append(Sewn(gc, design.stem, panels, part, look))
    return out


def _triangles(part: RestPart) -> IntArray:
    starts = np.concatenate([[0], np.cumsum(part.counts)[:-1]])
    fan = [(s, s + k, s + k + 1) for s, n in zip(starts, part.counts, strict=True) for k in range(1, n - 1)]
    return part.indices[np.asarray(fan, dtype=np.int64)]


def build_hero(recipe: Recipe, spec: HumanSpec, rest_file: Path, folder: Path) -> dict[str, object]:
    human = load_rest(rest_file)
    usd = folder / "usd"
    looks = {
        p.name: _look(p.kind, folder / "textures" / p.name, _material_file(p, spec.eyes, recipe), usd)
        for p in human.parts
    }
    regions = body_regions(human)
    parts = _parts(human)
    sewn = _sew(recipe, spec, human, folder)
    looks.update({s.part.name: s.look for s in sewn})
    body = next(p for p in human.parts if p.kind == "body")
    rest = {p.name: p.points[None] for p in human.parts}
    human_usd.write(usd / "rest.usdc", parts=parts, samples=rest, times=None, looks=looks, regions=regions)
    clips: dict[str, object] = {}
    frames = min([recipe.clip_frames] + [s.recipe.clip_frames for s in sewn])
    for category in CATEGORIES:
        clip = build_clip(human, category, spec.motions[category], frames, recipe.cmu)
        samples = dict(clip.samples)
        for item in sewn:
            scratch = folder / "build" / f"drape_{clip.name}_{item.part.name}"
            scratch.mkdir(parents=True, exist_ok=True)
            samples[item.part.name] = garments.drape(
                item.recipe,
                item.panels,
                (body.points, _triangles(body)),
                (clip.times, clip.samples["body"]),
                scratch=scratch,
            )
        file = usd / f"{clip.name}.usdc"
        clip_parts = parts + [s.part for s in sewn]
        human_usd.write(
            file, parts=clip_parts, samples=samples, times=clip.times, looks=looks, regions=regions
        )
        clips[clip.name] = _clip_record(clip, file.relative_to(folder).as_posix())
    record = manifest(spec, human, clips, recipe.licenses)
    record["parts"].update(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        {
            f"{human_usd.ROOT}/{s.part.name}": {"kind": "garment_sim", "source": f"GarmentCode {s.design}"}
            for s in sewn
        }
    )
    return record


def manifest(
    spec: HumanSpec, human: RestHuman, clips: dict[str, object], licenses: tuple[dict[str, str], ...]
) -> dict[str, object]:
    root = human_usd.ROOT
    return {
        "key": spec.key,
        "kind": "human",
        "rest": "usd/rest.usdc",
        "clips": clips,
        "skin_prim": f"{root}/body",
        "parts": {f"{root}/{p.name}": {"kind": p.kind, "source": Path(p.source).stem} for p in human.parts},
        "regions": {
            name: f"{root}/body/{name}"
            for name in ("scalp", "face", "brows", "beard", "ears", "lips", "body")
        },
        "region_family": human_usd.REGION_FAMILY,
        "rest_primvar": human_usd.REST_PRIMVAR,
        "phenotype": spec.phenotype,
        "face_targets": [[t, v] for t, v in spec.targets],
        "skin": spec.skin.id,
        "eyes": spec.eyes,
        "outfit": list(spec.outfit),
        "source": "MPFB/MakeHuman body, skin and clothes; CMU motion capture",
        "license": "; ".join(f"{x['source']}: {x['license']}" for x in licenses),
        "licenses": list(licenses),
        "redistributable": True,
    }


def build_library(recipe: Recipe, library: Path, only: tuple[int, ...] | None = None) -> list[Path]:
    catalog = cast("Catalog", json.loads(recipe.mpfb_targets.read_text()))
    indices = recipe.indices() if only is None else only
    specs = [sample(recipe, i, catalog) for i in indices]
    pending = [s for s in specs if not (library / s.key / "build" / "rest.npz").exists()]
    for spec in pending:
        (library / spec.key / "build").mkdir(parents=True, exist_ok=True)
    if pending:
        jobs = [_blender_job(recipe, s, library / s.key / "build" / "rest.npz") for s in pending]
        run_blender(recipe, jobs, library)
    out: list[Path] = []
    for spec in specs:
        folder = library / spec.key
        record = build_hero(recipe, spec, folder / "build" / "rest.npz", folder)
        files.write_text(folder / "hero.json", json.dumps(record, indent=1, sort_keys=True) + "\n")
        out.append(folder)
    return out
