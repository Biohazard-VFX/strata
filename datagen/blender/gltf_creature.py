"""Import an authored glTF animal (optionally with an animation from a second glTF) and export it.

Run: blender -b --factory-startup --python gltf_creature.py -- JOB.json
JOB.json: model (glTF file), animation (glTF file whose armature action drives the model's bones, or
null), action (action name to use; default the animation file's first), subdivide (Catmull-Clark
levels applied after skinning, for low-poly models), frames, resolution, output.
Actions loop, so any frame window is valid. With an armature, only meshes it deforms are exported.
Coincident vertices are welded first so subdivision does not tear texture seams apart.
"""

import json
import sys
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))

import creature_export


def _import(path: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    return [o for o in bpy.data.objects if o not in before]


def _loop(action: bpy.types.Action) -> None:
    curves = [
        c
        for layer in action.layers
        for strip in layer.strips
        for bag in strip.channelbags
        for c in bag.fcurves
    ]
    for curve in curves:
        if not any(m.type == "CYCLES" for m in curve.modifiers):
            curve.modifiers.new("CYCLES")


def _animate(model: list[bpy.types.Object], job: dict[str, object]) -> str | None:
    armatures = [o for o in model if o.type == "ARMATURE"]
    if not armatures:
        return None
    before = set(bpy.data.actions)
    if job.get("animation"):
        extra = _import(str(job["animation"]))
        actions = [a for a in bpy.data.actions if a not in before]
        bpy.ops.object.select_all(action="DESELECT")
        for obj in extra:
            obj.select_set(state=True)
        bpy.ops.object.delete()
    else:
        actions = list(bpy.data.actions)
    wanted = job.get("action")
    chosen = next((a for a in actions if wanted is None or a.name.startswith(str(wanted))), None)
    if chosen is None:
        return None
    _loop(chosen)
    for armature in armatures:
        armature.animation_data_create()
        armature.animation_data.action = chosen
        if chosen.slots:
            armature.animation_data.action_slot = chosen.slots[0]
    return chosen.name


def _deformed(obj: bpy.types.Object, model: list[bpy.types.Object]) -> bool:
    if not any(o.type == "ARMATURE" for o in model):
        return True
    return any(m.type == "ARMATURE" for m in obj.modifiers)


def main() -> None:
    job = json.loads(Path(sys.argv[-1]).read_text())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    model = _import(str(job["model"]))
    action = _animate(model, job)
    meshes = sorted((o for o in model if o.type == "MESH" and _deformed(o, model)), key=lambda o: o.name)
    for mesh in meshes:
        welded = bmesh.new()
        welded.from_mesh(mesh.data)
        bmesh.ops.remove_doubles(welded, verts=welded.verts, dist=1e-5)
        welded.to_mesh(mesh.data)
        welded.free()
        if int(job.get("subdivide", 0)):
            modifier = mesh.modifiers.new("strata_subdivide", "SUBSURF")
            modifier.levels = modifier.render_levels = int(job["subdivide"])
    notes = {
        "model": Path(str(job["model"])).name,
        "animation": Path(str(job["animation"])).name if job.get("animation") else None,
        "action": action,
        "subdivide": int(job.get("subdivide", 0)),
        "objects": [o.name for o in meshes],
    }
    creature_export.export(
        meshes, list(job["frames"]), Path(job["output"]), resolution=int(job["resolution"]), notes=notes
    )
    print("EXPORTED", job["output"], flush=True)


main()
