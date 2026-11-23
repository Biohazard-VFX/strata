"""strata-datagen: fetch and prepare assets, build seeded shots, render them with husk, run QA."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from strata_datagen import files, shotlock
from strata_datagen.library import build as library_build
from strata_datagen.library import catalogue, creatures, fetch, humans, morph
from strata_datagen.presets import PRESETS
from strata_datagen.qa import checks, contact
from strata_datagen.render import husk, ingest, pack
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import DEFAULT_FRAMES, ShotSpec

DEFAULT_HUSK = "husk"


def _specs(args: argparse.Namespace) -> list[ShotSpec]:
    specs = [PRESETS[name] for name in args.presets]
    specs += [ShotSpec(seed=seed) for seed in range(args.first_seed, args.first_seed + args.count)]
    return specs


def cmd_fetch(args: argparse.Namespace) -> None:
    for folder in args.folders:
        print(fetch.fetch(args.category, folder, args.dest))


def cmd_prepare(args: argparse.Namespace) -> None:
    library = library_build.prepare(library_build.Sources.load(args.sources), args.root)
    print(
        json.dumps(
            {
                "models": len(library.models),
                "surfaces": len(library.surfaces),
                "hdris": len(library.hdris),
                "heroes": len(library.heroes),
            }
        )
    )


def cmd_humans(args: argparse.Namespace) -> None:
    recipe = humans.Recipe.load(args.recipe)
    only = None if args.only is None else tuple(args.only)
    for folder in humans.build_library(recipe, args.library, only):
        print("built", folder, flush=True)


def cmd_morphs(args: argparse.Namespace) -> None:
    recipe = humans.Recipe.load(args.recipe)
    block = morph.MorphRecipe.parse(json.loads(args.recipe.read_text())["morph"])
    catalog = cast("humans.Catalog", json.loads(recipe.mpfb_targets.read_text()))
    for family in args.families if args.families is not None else range(block.families):
        print("built", morph.build_family(recipe, block, family, args.library, catalog), flush=True)


def cmd_creatures(args: argparse.Namespace) -> None:
    raw = cast("dict[str, list[dict[str, object]]]", json.loads(args.recipes.read_text()))
    recipes = [r for r in raw["creatures"] if not args.only or r["key"] in args.only]
    tools = creatures.Tools(infinigen=[args.infinigen_python], blender=[args.blender])
    for manifest in creatures.prepare(recipes, tools, args.library):
        print("creature", manifest["key"], ",".join(cast("list[str]", manifest["coat_classes"])), flush=True)


def cmd_build(args: argparse.Namespace) -> None:
    library = catalogue.load(args.root)
    render = compose.RenderParams(args.width, args.height, args.samples)
    jobs: list[husk.Job] = []
    for spec in _specs(args):
        shot = compose.plan(spec, library, args.root, args.frames)
        shot.directory.mkdir(parents=True, exist_ok=True)
        with shotlock.held(shot.directory):
            compose.write(shot, library, args.root, render)
            export.write(shot, library, render)
        jobs += husk.jobs(shot.directory, args.root, shot.frames, args.chunk)
        print("built", spec.shot_id, shot.track.move, shot.light.hdri.key, flush=True)
    husk.write_job_list(args.root, jobs)


def finish_task(shot: Path, frames: range) -> bool:
    """Pack and raw-QA every rendered frame, retaining all deep; False when raw QA fails."""
    pack.pack(shot, frames)
    return _accepted(shot, frames)


def render_task(shot: Path, frames: range, husk_path: str) -> dict[str, object]:
    """Render, pack and accept `frames` under one hold of the shot lock; husk inherits it."""
    with shotlock.held(shot) as hold:
        timings = husk.run(shot, frames, husk_path, hold.descriptor)
        if timings["exit_code"] == 0:
            timings["post_passed"] = finish_task(shot, frames)
        manifest_path = shot / "manifest.json"
        manifest = export.load_json(manifest_path)
        manifest["timings"] = timings
        files.write_text(manifest_path, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return timings


def _passed(timings: dict[str, object]) -> bool:
    return timings["exit_code"] == 0 and bool(timings["post_passed"])


def cmd_render_task(args: argparse.Namespace) -> None:
    timings = render_task(args.shot, _frames(args), args.husk)
    print("rendered", timings["device"], timings["mean_frame_seconds"], flush=True)
    if not _passed(timings):
        raise SystemExit(1)


def cmd_render(args: argparse.Namespace) -> None:
    """Run every selected task, then exit non-zero naming the tasks that failed to render or pass QA."""
    jobs = [husk.Job(**j) for j in json.loads((args.root / "jobs.json").read_text())]
    failed: list[str] = []
    for job in jobs:
        if args.shots and job.shot_id not in args.shots:
            continue
        frames = range(job.first, job.first + job.count)
        timings = render_task(args.root / job.workdir, frames, args.husk)
        print("rendered", job.shot_id, timings["device"], timings["mean_frame_seconds"], flush=True)
        if not _passed(timings):
            failed.append(f"{job.shot_id}:{job.first}-{job.first + job.count - 1}")
    if failed:
        print("failed", *failed, flush=True)
        raise SystemExit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    print("ingested", len(ingest.ingest(args.shot, args.source)), "into", args.shot, flush=True)


def _frames(args: argparse.Namespace) -> range:
    return range(args.first, args.first + args.count)


def cmd_pack(args: argparse.Namespace) -> None:
    pack.pack(args.shot, _frames(args))
    print("packed", args.first, args.count, flush=True)


def _accepted(shot: Path, frames: range) -> bool:
    failed = [r for r in checks.accept(shot, frames) if r.failures]
    for result in failed:
        print("frame", result.frame, "FAIL", result.failures[:3], flush=True)
    return not failed


def cmd_accept(args: argparse.Namespace) -> None:
    if not _accepted(args.shot, _frames(args)):
        raise SystemExit(1)
    print("accepted", args.first, args.count, flush=True)


def cmd_remove_samples(args: argparse.Namespace) -> None:
    print("removed sidecars", pack.remove_samples(args.shot, _frames(args)), flush=True)


def _label(shot: Path) -> str:
    scene = cast("dict[str, dict[str, object]]", export.load_json(shot / "manifest.json")["scene"])
    cam = cast("dict[str, float | str | bool]", scene["camera"])
    flags = "".join(f for f, on in (("MB ", cam["motion_blur"]), ("DOF ", cam["depth_of_field"])) if on)
    return f"{shot.name}  {cam['move']}  {cam['focal_length_mm']:.0f}mm  {flags}"


def _review(shot: Path, rejected: Path) -> tuple[Path, dict[str, object], str]:
    with shotlock.held(shot) as hold:
        result = checks.check_shot(shot)
        files.write_text(shot / "qa.json", json.dumps(result, indent=1) + "\n")
        label = _label(shot)
        if not result["passed"]:
            rejected.mkdir(exist_ok=True)
            hold.move(rejected / shot.name)
        return hold.location, result, label


def cmd_qa(args: argparse.Namespace) -> None:
    shots = sorted(p for p in (args.root / "shots").iterdir() if (p / "manifest.json").exists())
    rejected = args.root / "rejected"
    reviews = [_review(shot, rejected) for shot in shots]
    contact.build([(location, label) for location, _, label in reviews], args.root / "contact_sheet.jpg")
    results = [result for _, result, _ in reviews]
    report = [{k: r[k] for k in ("shot_id", "passed", "failures", "mean_metrics")} for r in results]
    files.write_text(args.root / "qa_report.json", json.dumps(report, indent=1) + "\n")
    for shot, result in zip(shots, results, strict=True):
        failures = cast("list[str]", result["failures"])
        print(shot.name, "PASS" if result["passed"] else "FAIL", failures[:3], flush=True)


def _hero_library_commands(add_parser: Callable[..., argparse.ArgumentParser]) -> None:
    p = add_parser("humans", help="build the human hero library from a recipe (MPFB + CMU)")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--only", type=int, nargs="*", help="human indices to build (default: all)")
    p.set_defaults(run=cmd_humans)
    p = add_parser("morphs", help="build morphing human takes (recipe block `morph`)")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--families", type=int, nargs="*", help="family indices to build (default: all)")
    p.set_defaults(run=cmd_morphs)
    p = add_parser("creatures", help="generate or import creature bodies into a hero library")
    p.add_argument("--recipes", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--infinigen-python", required=True)
    p.add_argument("--blender", default="blender")
    p.add_argument("--only", nargs="*", default=[])
    p.set_defaults(run=cmd_creatures)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="strata-datagen", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="download Megascans folders that are not synced locally")
    p.add_argument("category")
    p.add_argument("folders", nargs="+")
    p.add_argument("--dest", type=Path, required=True)
    p.set_defaults(run=cmd_fetch)
    p = sub.add_parser("prepare", help="build the asset library of a build root")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.set_defaults(run=cmd_prepare)
    _hero_library_commands(sub.add_parser)
    p = sub.add_parser("build", help="compose shots and write the husk job list")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--presets", nargs="*", default=[], choices=sorted(PRESETS))
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument(
        "--frames", type=int, default=DEFAULT_FRAMES, help="frames per shot (presets fix their own)"
    )
    p.add_argument("--chunk", type=int, default=husk.CHUNK_FRAMES, help="frames per farm task")
    p.set_defaults(run=cmd_build)
    p = sub.add_parser("render", help="run the job list's render tasks locally")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shots", nargs="*", default=[])
    p.add_argument("--husk", default=DEFAULT_HUSK)
    p.set_defaults(run=cmd_render)
    p = sub.add_parser("render-task", help="one job: render with husk, pack and accept, all under the lock")
    p.add_argument("--shot", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--husk", default=DEFAULT_HUSK)
    p.set_defaults(run=cmd_render_task)
    p = sub.add_parser(
        "ingest", help="move a downloaded farm task's output folders into a shot under its lock"
    )
    p.add_argument("--shot", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.set_defaults(run=cmd_ingest)
    for name, run, text in (
        ("pack", cmd_pack, "write layer targets and raw/merged sidecars for every rendered frame"),
        ("accept", cmd_accept, "raw QA of packed frames; records each frame's acceptance"),
        ("remove-samples", cmd_remove_samples, "offline, no reader running: delete published sidecars"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--shot", type=Path, required=True)
        p.add_argument("--first", type=int, required=True)
        p.add_argument("--count", type=int, required=True)
        p.set_defaults(run=run)
    p = sub.add_parser("qa", help="check every shot, reject failures, write the report and contact sheet")
    p.add_argument("--root", type=Path, required=True)
    p.set_defaults(run=cmd_qa)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.run(args)
