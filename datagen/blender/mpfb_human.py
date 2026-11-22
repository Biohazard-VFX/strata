"""Build MPFB humans in Blender and export their rest geometry, skin weights and rig to .npz files.

Run: blender -b --python mpfb_human.py -- jobs.json (MPFB enabled in the Blender user preferences).
jobs.json: [{"recipe": {...}, "output": "human.npz"}]; a recipe holds phenotype (MPFB macro values),
targets ([{"target", "value"}]), rig (MPFB built-in rig name), eyes (mhclo) and clothes (mhclo list).
Coordinates are Blender world space (Z up, metres); the importer converts them.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

REGION_GROUPS = ("scalp", "ears", "lips", "fingernails", "toenails")
INFLUENCES = 4


Part = tuple[str, str, bpy.types.Object, str]


def _services() -> tuple[type, type]:
    package = next(name for name in bpy.context.preferences.addons if name.endswith("mpfb"))
    humans = __import__(package + ".services.humanservice", fromlist=["HumanService"])
    targets = __import__(package + ".services.targetservice", fromlist=["TargetService"])
    return humans.HumanService, targets.TargetService


def _build(recipe: dict[str, object]) -> tuple[bpy.types.Object, list[Part]]:
    human_service, target_service = _services()
    bpy.ops.wm.read_homefile(use_empty=True)
    basemesh = human_service.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=recipe["phenotype"],
    )
    if recipe["targets"]:
        target_service.bulk_load_targets(basemesh, recipe["targets"])
    bpy.context.view_layer.update()
    human_service.add_builtin_rig(basemesh, recipe["rig"], import_weights=True)
    parts = [("body", "body", basemesh, recipe["skin"])]
    eyes = human_service.add_mhclo_asset(
        recipe["eyes"], basemesh, asset_type="eyes", subdiv_levels=0, material_type="NONE"
    )
    parts.append(("eyes", "eyes", eyes, recipe["eyes"]))
    for index, mhclo in enumerate(recipe["clothes"]):
        obj = human_service.add_mhclo_asset(
            mhclo, basemesh, asset_type="clothes", subdiv_levels=0, material_type="NONE"
        )
        parts.append((f"garment{index}", "garment", obj, mhclo))
    armature = basemesh.parent
    armature.data.pose_position = "REST"
    for _, _, obj, _ in parts:
        for modifier in obj.modifiers:
            if modifier.type == "SUBSURF":
                modifier.show_render = modifier.show_viewport = False
    bpy.context.view_layer.update()
    return armature, parts


def _bones(armature: bpy.types.Object) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bones = list(armature.data.bones)
    names = [b.name for b in bones]
    world = np.array(armature.matrix_world)
    rest = np.stack([world @ np.array(b.matrix_local) for b in bones])
    heads = np.stack([(world @ np.append(np.array(b.head_local), 1.0))[:3] for b in bones])
    tails = np.stack([(world @ np.append(np.array(b.tail_local), 1.0))[:3] for b in bones])
    parents = np.array([names.index(b.parent.name) if b.parent else -1 for b in bones], dtype=np.int32)
    return names, rest, heads, tails, parents


def _mesh(obj: bpy.types.Object, bone_names: list[str]) -> dict[str, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    world = np.array(obj.matrix_world)
    count = len(mesh.vertices)
    co = np.empty(count * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    counts = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", counts)
    indices = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", indices)
    uv = np.empty(len(mesh.loops) * 2)
    mesh.uv_layers.active.data.foreach_get("uv", uv)
    group_names = {g.index: g.name for g in obj.vertex_groups}
    bone_index = {name: i for i, name in enumerate(bone_names)}
    joints = np.zeros((count, INFLUENCES), dtype=np.int32)
    weights = np.zeros((count, INFLUENCES))
    regions = {name: np.zeros(count, dtype=bool) for name in REGION_GROUPS}
    for vertex in mesh.vertices:
        influences = []
        for element in vertex.groups:
            name = group_names.get(element.group)
            if name in bone_index and element.weight > 0.0:
                influences.append((element.weight, bone_index[name]))
            elif name in regions and element.weight > 0.5:
                regions[name][vertex.index] = True
        influences.sort(key=lambda pair: (-pair[0], pair[1]))
        for slot, (weight, bone) in enumerate(influences[:INFLUENCES]):
            joints[vertex.index, slot] = bone
            weights[vertex.index, slot] = weight
    evaluated.to_mesh_clear()
    total = weights.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise RuntimeError(f"{obj.name}: {int(np.sum(total <= 0.0))} vertices have no bone weights")
    return {
        "points": co,
        "counts": counts,
        "indices": indices,
        "uv": uv.reshape(-1, 2),
        "joints": joints,
        "weights": weights / total,
        **{f"region_{k}": v for k, v in regions.items()},
    }


def _export(armature: bpy.types.Object, parts: list[Part], output: Path) -> None:
    names, rest, heads, tails, parents = _bones(armature)
    arrays = {"bone_rest": rest, "bone_heads": heads, "bone_tails": tails, "bone_parents": parents}
    meta = {"bones": names, "parts": []}
    for name, kind, obj, source in parts:
        for key, value in _mesh(obj, names).items():
            if key.startswith("region_") and kind != "body":
                continue
            arrays[f"{name}/{key}"] = value
        meta["parts"].append({"name": name, "kind": kind, "source": source})
    arrays["meta"] = np.array(json.dumps(meta))
    temporary = output.with_name(f".tmp-{output.name}")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)


def main() -> None:
    jobs = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    for job in jobs:
        armature, parts = _build(job["recipe"])
        _export(armature, parts, Path(job["output"]))
        print("exported", job["output"], flush=True)


main()
