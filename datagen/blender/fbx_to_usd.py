"""Convert FBX models to USD (meshes, uvs, normals only), one Blender process for many files.

Run: blender -b --factory-startup --python fbx_to_usd.py -- JOBS.json
JOBS.json is a list of [input.fbx, output.usdc] pairs; outputs are Y-up and in metres.
"""

import json
import sys
from pathlib import Path

import bpy


def convert(source: Path, target: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(target),
        selected_objects_only=False,
        export_materials=False,
        export_uvmaps=True,
        export_normals=True,
        export_animation=False,
        export_armatures=False,
        convert_orientation=True,
        export_global_forward_selection="NEGATIVE_Z",
        export_global_up_selection="Y",
    )


def main() -> None:
    jobs = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    for source, target in jobs:
        convert(Path(source), Path(target))
        print("CONVERTED", target, flush=True)


main()
