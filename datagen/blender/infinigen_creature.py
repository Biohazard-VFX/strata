"""Generate one Infinigen creature (hairless body, optionally animated) and export it for the hero library.

Run with Infinigen's Python (bpy as a module):
  python infinigen_creature.py JOB.json
JOB.json: factory (creature factory class name, or module.Class inside infinigen's creatures package
for factories it does not export), factory_seed, factory_kwargs, asset_kwargs,
nurbs (prefix -> exact Infinigen NURBS template key, to pin a body or head build), frames, resolution,
output (folder).
"""

import importlib
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import creature_export
from infinigen.assets.objects import creatures
from infinigen.assets.objects.creatures.parts import generic_nurbs
from infinigen.core.util.math import FixedSeed

SKIP = ("placeholder", "cube", "foot_path", "ik_target")


def _pin_templates(nurbs: dict[str, str]) -> None:
    original = generic_nurbs.NurbsPart.__init__

    def pinned(
        self: generic_nurbs.NurbsPart, *args: object, prefix: str | None = None, **kwargs: object
    ) -> None:
        original(self, *args, prefix=nurbs.get(prefix or "", prefix), **kwargs)

    generic_nurbs.NurbsPart.__init__ = pinned


def _meshes(root: bpy.types.Object) -> list[bpy.types.Object]:
    found = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and not any(word in o.name.lower() for word in SKIP) and _under(o, root)
    ]
    return sorted(found, key=lambda o: o.name)


def _under(obj: bpy.types.Object, root: bpy.types.Object) -> bool:
    parent = obj.parent
    while parent is not None:
        if parent == root:
            return True
        parent = parent.parent
    return False


def main() -> None:
    job = json.loads(Path(sys.argv[-1]).read_text())
    _pin_templates(job.get("nurbs", {}))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    module, _, name = job["factory"].rpartition(".")
    package = importlib.import_module(f"{creatures.__name__}.{module}") if module else creatures
    factory_class = getattr(package, name)
    with FixedSeed(job["factory_seed"]):
        factory = factory_class(factory_seed=job["factory_seed"], **job.get("factory_kwargs", {}))
        placeholder = factory.spawn_placeholder(0, (0, 0, 0), (0, 0, 0))
        root = factory.spawn_asset(0, placeholder=placeholder, **job.get("asset_kwargs", {}))
    top = root
    while top.parent is not None:
        top = top.parent
    meshes = _meshes(top)
    notes = {
        "factory": job["factory"],
        "factory_seed": job["factory_seed"],
        "factory_kwargs": job.get("factory_kwargs", {}),
        "asset_kwargs": job.get("asset_kwargs", {}),
        "nurbs": job.get("nurbs", {}),
        "objects": [o.name for o in meshes],
    }
    creature_export.export(
        meshes, list(job["frames"]), Path(job["output"]), resolution=int(job["resolution"]), notes=notes
    )
    print("EXPORTED", job["output"], flush=True)


main()
