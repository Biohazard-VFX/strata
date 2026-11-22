"""Shared Blender-side export for creature bodies: one skin with regions, baked albedo, per-frame points.

Imported by the creature scripts in this folder; runs inside Blender's Python (bpy), never in datagen.
Output folder: skin.npz (points per frame in Blender world space, faces, corner uvs, face regions),
albedo.png, creature.json (region names, frames, source notes).
"""

import json
from collections.abc import Callable
from pathlib import Path

import bmesh
import bpy
import numpy as np

REGION_TAGS = ("head", "jaw", "head_detail", "body", "tail", "leg", "foot", "wing")
EXTRA_REGIONS = (
    ("eyeball", "eye"),
    ("eye", "eye"),
    ("teeth", "teeth"),
    ("tongue", "tongue"),
    ("claw", "claw"),
    ("nose", "nose"),
    ("lid", "lid"),
    ("beak", "beak"),
    ("feather", "feather"),
    ("horn", "horn"),
    ("hoof", "foot"),
)


def _region_names() -> list[str]:
    names = list(REGION_TAGS)
    for _, region in EXTRA_REGIONS:
        if region not in names:
            names.append(region)
    return [*names, "other"]


REGIONS = _region_names()


def _face_regions(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> np.ndarray:
    tags = [t for t in REGION_TAGS if f"tag_{t}" in mesh.attributes]
    faces = len(mesh.polygons)
    if tags:
        scores = np.zeros((faces, len(tags)))
        for column, tag in enumerate(tags):
            attribute = mesh.attributes[f"tag_{tag}"]
            values = np.zeros(len(attribute.data))
            attribute.data.foreach_get("value", values)
            if attribute.domain == "POINT":
                corner_vert = np.zeros(len(mesh.loops), dtype=np.int64)
                mesh.loops.foreach_get("vertex_index", corner_vert)
                starts = np.zeros(faces, dtype=np.int64)
                mesh.polygons.foreach_get("loop_start", starts)
                counts = np.zeros(faces, dtype=np.int64)
                mesh.polygons.foreach_get("loop_total", counts)
                summed = np.add.reduceat(values[corner_vert], starts)
                values = summed / counts
            scores[:, column] = values
        best = np.argmax(scores, axis=1)
        ids = np.array([REGIONS.index(t) for t in tags])[best]
        ids[scores.max(axis=1) <= 0.0] = REGIONS.index("body")
        return ids
    name = obj.name.lower()
    for word, region in EXTRA_REGIONS:
        if word in name:
            return np.full(faces, REGIONS.index(region))
    return np.full(faces, REGIONS.index("body"))


def _applied_copy(obj: bpy.types.Object, index: int) -> bpy.types.Object:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), depsgraph=depsgraph)
    mesh.transform(obj.matrix_world)
    copy = bpy.data.objects.new(f"bake_{index}", mesh)
    bpy.context.scene.collection.objects.link(copy)
    regions = _face_regions(obj, mesh)
    attribute = mesh.attributes.new("strata_region", "INT", "FACE")
    attribute.data.foreach_set("value", regions.astype(np.int32))
    source = mesh.attributes.new("strata_source", "INT", "POINT")
    ids = index * 10_000_000 + np.arange(len(mesh.vertices), dtype=np.int64)
    source.data.foreach_set("value", ids.astype(np.int32))
    return copy


def _join(copies: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for copy in copies:
        copy.select_set(state=True)
    bpy.context.view_layer.objects.active = copies[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    bm = bmesh.new()
    bm.from_mesh(joined.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.calc_area() <= 0.0], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(joined.data)
    bm.free()
    return joined


def _uv(joined: bpy.types.Object) -> None:
    layer = joined.data.uv_layers.new(name="strata_bake")
    joined.data.uv_layers.active = layer
    bpy.ops.object.select_all(action="DESELECT")
    joined.select_set(state=True)
    bpy.context.view_layer.objects.active = joined
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")


def _bake(joined: bpy.types.Object, path: Path, resolution: int) -> None:
    image = bpy.data.images.new("strata_albedo", resolution, resolution)
    if not joined.data.materials:
        material = bpy.data.materials.new("strata_default")
        material.use_nodes = True
        joined.data.materials.append(material)
    for material in joined.data.materials:
        if material is None or not material.use_nodes:
            continue
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        material.node_tree.nodes.active = node
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 4
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.margin = 6
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()


def _topology(mesh: bpy.types.Mesh) -> dict[str, np.ndarray]:
    faces, corners = len(mesh.polygons), len(mesh.loops)
    counts = np.zeros(faces, dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", counts)
    indices = np.zeros(corners, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", indices)
    uv = np.zeros(corners * 2, dtype=np.float32)
    mesh.uv_layers["strata_bake"].data.foreach_get("uv", uv)
    regions = np.zeros(faces, dtype=np.int32)
    mesh.attributes["strata_region"].data.foreach_get("value", regions)
    source = np.zeros(len(mesh.vertices), dtype=np.int32)
    mesh.attributes["strata_source"].data.foreach_get("value", source)
    return {
        "counts": counts,
        "indices": indices,
        "uv": uv.reshape(-1, 2),
        "regions": regions,
        "source": source,
    }


def _world_points(obj: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = np.zeros(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", points)
    evaluated.to_mesh_clear()
    matrix = np.array(obj.matrix_world)
    local = points.reshape(-1, 3)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def export(
    objects: list[bpy.types.Object],
    frames: list[int],
    output: Path,
    *,
    resolution: int,
    notes: dict[str, object],
    before_frame: Callable[[int], None] | None = None,
) -> None:
    """Write skin.npz, albedo.png and creature.json for `objects` evaluated at `frames`."""
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.frame_set(frames[0])
    copies = [_applied_copy(obj, index) for index, obj in enumerate(objects)]
    joined = _join(copies)
    _uv(joined)
    topology = _topology(joined.data)
    _bake(joined, output / "albedo.png", resolution)
    source = topology.pop("source").astype(np.int64)
    owner, vertex = source // 10_000_000, source % 10_000_000
    points = np.zeros((len(frames), len(source), 3), dtype=np.float32)
    for row, frame in enumerate(frames):
        if before_frame is not None:
            before_frame(frame)
        scene.frame_set(frame)
        for index, obj in enumerate(objects):
            chosen = owner == index
            points[row, chosen] = _world_points(obj)[vertex[chosen]]
    np.savez_compressed(output / "skin.npz", points=points, **topology)
    info = {"regions": REGIONS, "frames": frames, "up": "z", **notes}
    (output / "creature.json").write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
