"""GarmentCode garments simulated with Houdini Vellum on human clips (a starter set of heroes).

Sewing patterns come from GarmentCode (MIT code, run in its own Python) for measurements taken
from the MPFB body; garments/garmentcode_panels.py exports the flat panels and seam pairs and
houdini/vellum_drape.py sews and drapes them on the animated body. GarmentCode's own simulator
(a non-commercial Warp fork) is never used.
"""

import itertools
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from strata_datagen import files
from strata_datagen.usdutil import FloatArray, IntArray

REPO = Path(__file__).resolve().parents[2]
PANELS_SCRIPT = REPO / "garments" / "garmentcode_panels.py"
DRAPE_SCRIPT = REPO / "houdini" / "vellum_drape.py"
CM = 100.0
SIM_STEPS_PER_FRAME = 2
PREROLL = 48
SETTLE = 24
SLICE_M = 0.01
ARM_GAP_M = 0.03


@dataclass(frozen=True)
class GarmentRecipe:
    """Recipe block `garmentcode`: tools, measurement templates and designs per sex."""

    python: str
    repo: Path
    hython: str
    templates: dict[str, Path]
    """GarmentCode measurement templates per sex, as JSON {"body": {name: cm}}."""
    designs: dict[str, tuple[Path, ...]]
    resolution_cm: float
    first_index: int
    count: int
    clip_frames: int

    @staticmethod
    def parse(raw: dict[str, object]) -> "GarmentRecipe":
        designs = cast("dict[str, list[str]]", raw["designs"])
        return GarmentRecipe(
            python=str(raw["python"]),
            repo=Path(str(raw["repo"])),
            hython=str(raw["hython"]),
            templates={k: Path(v) for k, v in cast("dict[str, str]", raw["templates"]).items()},
            designs={k: tuple(Path(p) for p in v) for k, v in designs.items()},
            resolution_cm=float(cast("float", raw["resolution_cm"])),
            first_index=int(cast("int", raw["first_index"])),
            count=int(cast("int", raw["count"])),
            clip_frames=int(cast("int", raw["clip_frames"])),
        )

    def applies(self, index: int) -> bool:
        return self.first_index <= index < self.first_index + self.count


def girth(points: FloatArray, height: float) -> float:
    """Perimeter around the torso slice at `height`; arms are split off at gaps along x."""
    near = points[np.abs(points[:, 1] - height) < SLICE_M]
    xs = np.sort(near[:, 0])
    cuts = xs[:-1][np.diff(xs) > ARM_GAP_M] + 0.5 * np.diff(xs)[np.diff(xs) > ARM_GAP_M]
    low = cuts[cuts < 0.0].max(initial=-np.inf)
    high = cuts[cuts > 0.0].min(initial=np.inf)
    torso = near[(near[:, 0] > low) & (near[:, 0] < high)]
    xz = torso[:, [0, 2]]
    centre = xz.mean(axis=0)
    angle = np.arctan2(xz[:, 1] - centre[1], xz[:, 0] - centre[0])
    bins = np.linspace(-np.pi, np.pi, 49)
    rim = [xz[(angle >= a) & (angle < b)] for a, b in itertools.pairwise(bins)]
    far = np.array([r[np.argmax(np.linalg.norm(r - centre, axis=1))] for r in rim if len(r)])
    return float(np.linalg.norm(np.diff(np.vstack([far, far[:1]]), axis=0), axis=1).sum())


def measure(body: FloatArray, heads: dict[str, FloatArray], template: dict[str, float]) -> dict[str, float]:
    """GarmentCode body measurements in cm.

    Lengths are scaled from the template by height; girths and levels are measured on the body.
    """
    height = float(body[:, 1].max())
    scale = height * CM / template["height"]
    out = {k: v * scale for k, v in template.items()}
    neck = float(heads["Neck"][1])
    hips_y = float(heads["LeftUpLeg"][1])
    chest_y = float(heads["LeftShoulder"][1]) - 0.12
    levels = np.arange(hips_y, chest_y, SLICE_M)
    girths = np.array([girth(body, y) for y in levels])
    waist = int(np.argmin(girths[len(levels) // 4 :])) + len(levels) // 4
    hips = int(np.argmax(girths[:waist]))
    bust = int(np.argmax(girths[waist:])) + waist
    arm = heads["LeftForeArm"] - heads["LeftArm"]
    out.update(
        height=height * CM,
        head_l=(height - neck) * CM,
        waist=girths[waist] * CM,
        hips=girths[hips] * CM,
        bust=girths[bust] * CM,
        waist_line=(neck - levels[waist]) * CM,
        hips_line=(levels[waist] - levels[hips]) * CM,
        arm_pose_angle=float(np.degrees(np.arctan2(-arm[1], abs(arm[0])))),
    )
    return {k: round(float(v), 4) for k, v in out.items()}


def write_measurements(path: Path, measurements: dict[str, float]) -> None:
    """JSON is valid YAML, which GarmentCode's body loader reads."""
    files.write_text(path, json.dumps({"body": measurements}, indent=1, sort_keys=True))


def run_panels(recipe: GarmentRecipe, jobs: list[dict[str, object]], scratch: Path) -> None:
    listing = scratch / "garmentcode_jobs.json"
    files.write_text(listing, json.dumps(jobs))
    subprocess.run(
        [recipe.python, str(PANELS_SCRIPT), str(listing)],
        check=True,
        capture_output=True,
        timeout=3600,
        cwd=recipe.repo,
        env={"PYTHONPATH": str(recipe.repo), "PATH": "/usr/bin:/bin"},
    )


def resample(times: FloatArray, values: FloatArray, at: FloatArray) -> FloatArray:
    """Values (len(times), n, 3) linearly interpolated at sorted times `at` inside the range."""
    upper = np.clip(np.searchsorted(times, at), 1, len(times) - 1)
    t0, t1 = times[upper - 1], times[upper]
    w = ((at - t0) / (t1 - t0))[:, None, None]
    return values[upper - 1] * (1.0 - w) + values[upper] * w


def sim_grid(times: FloatArray) -> FloatArray:
    """Simulation times: SIM_STEPS_PER_FRAME steps per frame from the first clip sample."""
    return np.arange(times[0], times[-1] + 1e-9, 1.0 / SIM_STEPS_PER_FRAME)


def preroll(rest: FloatArray, first: FloatArray) -> FloatArray:
    """Rest pose moved onto the first clip sample, blended into it, then held while cloth settles."""
    moved = rest + (first.mean(axis=0) - rest.mean(axis=0)) * np.array([1.0, 0.0, 1.0])
    w = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, PREROLL))[:, None, None]
    blend = moved[None] * (1.0 - w) + first[None] * w
    return np.concatenate(
        [np.repeat(moved[None], SETTLE // 2, axis=0), blend, np.repeat(first[None], SETTLE, axis=0)]
    )


def drape(
    recipe: GarmentRecipe,
    panels: Path,
    body: tuple[FloatArray, IntArray],
    clip: tuple[FloatArray, FloatArray],
    *,
    scratch: Path,
) -> FloatArray:
    """Cloth points at every clip sample time: (len(times), points, 3)."""
    rest, triangles = body
    times, samples = clip
    grid = sim_grid(times)
    steps = resample(times, samples, grid)
    roll = preroll(rest, samples[0])
    shift = (samples[0].mean(axis=0) - rest.mean(axis=0)) * np.array([1.0, 0.0, 1.0])
    moved_panels = scratch / f"{panels.stem}_moved.npz"
    with np.load(panels) as raw:
        data = {k: raw[k] for k in raw.files}
    data["points"] = data["points"] + shift
    np.savez_compressed(moved_panels, **data)
    body_file = scratch / "body_frames.npz"
    np.savez_compressed(
        body_file, frames=np.concatenate([roll, steps]).astype(np.float32), triangles=triangles
    )
    job = {
        "panels": str(moved_panels),
        "body": str(body_file),
        "output": str(scratch / "cloth.npz"),
        "fps": 24.0 * SIM_STEPS_PER_FRAME,
        "record_from": len(roll),
    }
    files.write_text(scratch / "drape_job.json", json.dumps(job))
    subprocess.run(
        [recipe.hython, str(DRAPE_SCRIPT), str(scratch / "drape_job.json")],
        check=True,
        capture_output=True,
        timeout=7200,
    )
    with np.load(scratch / "cloth.npz") as cloth:
        points = cloth["points"].astype(np.float64)
    return resample(grid, points, times)


def fabric_texture(path: Path, rng: np.random.Generator) -> Path:
    """A plain fabric colour as a small sRGB texture, drawn from muted garment colours."""
    hsv = (float(rng.uniform(0.0, 1.0)), float(rng.uniform(0.15, 0.7)), float(rng.uniform(0.2, 0.85)))
    image = Image.new("HSV", (8, 8), tuple(round(255 * v) for v in hsv)).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with files.replacing(path) as temporary:
        image.save(temporary, format="JPEG", quality=95)
    return path
