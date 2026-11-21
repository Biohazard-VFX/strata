"""strata-datagen: fetch and prepare assets, build seeded shots, render them with husk, run QA."""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from strata_datagen.library import build as library_build
from strata_datagen.library import catalogue, fetch
from strata_datagen.presets import PRESETS
from strata_datagen.qa import checks, contact
from strata_datagen.render import husk
from strata_datagen.scene import compose, export
from strata_datagen.scene.spec import FRAMES, ShotSpec

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


def cmd_build(args: argparse.Namespace) -> None:
    library = catalogue.load(args.root)
    render = compose.RenderParams(args.width, args.height, args.samples)
    jobs: list[husk.Job] = []
    for spec in _specs(args):
        shot = compose.plan(spec, library, args.root)
        compose.write(shot, library, args.root, render)
        export.write(shot, library, render)
        jobs.append(husk.job(args.root, shot.directory, FRAMES))
        print("built", spec.shot_id, shot.track.move, shot.light.hdri.key, flush=True)
    husk.write_job_list(args.root, jobs)


def cmd_render(args: argparse.Namespace) -> None:
    jobs = [husk.Job(**j) for j in json.loads((args.root / "jobs.json").read_text())]
    for job in jobs:
        if args.shots and job.shot_id not in args.shots:
            continue
        timings = husk.run(args.root, job, args.husk)
        manifest_path = args.root / job.workdir / "manifest.json"
        manifest = export.load_json(manifest_path)
        manifest["timings"] = timings
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print("rendered", job.shot_id, timings["device"], timings["mean_frame_seconds"], flush=True)


def _label(shot: Path) -> str:
    scene = cast("dict[str, dict[str, object]]", export.load_json(shot / "manifest.json")["scene"])
    cam = cast("dict[str, float | str | bool]", scene["camera"])
    flags = "".join(f for f, on in (("MB ", cam["motion_blur"]), ("DOF ", cam["depth_of_field"])) if on)
    return f"{shot.name}  {cam['move']}  {cam['focal_length_mm']:.0f}mm  {flags}"


def cmd_qa(args: argparse.Namespace) -> None:
    shots = sorted(p for p in (args.root / "shots").iterdir() if (p / "manifest.json").exists())
    results = [checks.check_shot(shot) for shot in shots]
    for shot, result in zip(shots, results, strict=True):
        (shot / "qa.json").write_text(json.dumps(result, indent=1) + "\n")
    contact.build([(s, _label(s)) for s in shots], args.root / "contact_sheet.jpg")
    report = [{k: r[k] for k in ("shot_id", "passed", "failures", "mean_metrics")} for r in results]
    (args.root / "qa_report.json").write_text(json.dumps(report, indent=1) + "\n")
    rejected = args.root / "rejected"
    for shot, result in zip(shots, results, strict=True):
        if not result["passed"]:
            rejected.mkdir(exist_ok=True)
            shutil.move(shot, rejected / shot.name)
        failures = cast("list[str]", result["failures"])
        print(shot.name, "PASS" if result["passed"] else "FAIL", failures[:3], flush=True)


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
    p = sub.add_parser("build", help="compose shots and write the husk job list")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--presets", nargs="*", default=[], choices=sorted(PRESETS))
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=64)
    p.set_defaults(run=cmd_build)
    p = sub.add_parser("render", help="render the job list locally")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shots", nargs="*", default=[])
    p.add_argument("--husk", default=DEFAULT_HUSK)
    p.set_defaults(run=cmd_render)
    p = sub.add_parser("qa", help="check every shot, reject failures, write the report and contact sheet")
    p.add_argument("--root", type=Path, required=True)
    p.set_defaults(run=cmd_qa)
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.run(args)
