"""Drape and simulate GarmentCode panels on an animated body with Houdini Vellum, run in hython.

Run: hython vellum_drape.py job.json. job.json: {"panels": panels.npz (from garmentcode_panels.py),
"body": body.npz with frames (F, n, 3) and triangles (m, 3), "output": cloth.npz,
"fps": simulation frames per second, "record_from": first body frame to record}. Body frames before
`record_from` are the pre-roll that sews the panels and settles the cloth. The output holds cloth
points (F - record_from, p, 3) and the panel triangles.
"""

import json
import sys
from pathlib import Path

import hou
import numpy as np

SOURCE = """
import hou
node = hou.pwd()
geo = node.geometry()
data = hou.session.strata_drape[node.name()]
frame = min(max(int(round(hou.frame())) - 1, 0), len(data["frames"]) - 1)
geo.createPoints(data["frames"][frame].tolist())
geo.createPolygons(data["polygons"], True)
if "lines" in data:
    lines = geo.createPolygons(data["lines"], False)
    group = geo.createPrimGroup("seams")
    group.add(lines)
    panels = geo.createPrimGroup("panels")
    panels.add([p for p in geo.prims() if p.type() == hou.primType.Polygon and p.isClosed()])
"""


def _python_sop(parent: hou.Node, name: str) -> hou.Node:
    node = parent.createNode("python", name)
    node.parm("python").set(SOURCE)
    return node


def _network() -> hou.Node:
    geo = hou.node("/obj").createNode("geo", "drape")
    cloth = _python_sop(geo, "cloth")
    body = _python_sop(geo, "body")
    fabric = geo.createNode("vellumconstraints", "fabric")
    fabric.setInput(0, cloth)
    fabric.parm("constrainttype").set("cloth")
    fabric.parm("group").set("panels")
    fabric.parm("dothickness").set("on")
    fabric.parm("thickness").set(0.004)
    fabric.parm("bendstiffness").set(0.2)
    seams = geo.createNode("vellumconstraints", "seams")
    seams.setInput(0, fabric, 0)
    seams.setInput(1, fabric, 1)
    seams.parm("constrainttype").set("distance")
    seams.parm("group").set("seams")
    seams.parm("stretchrestscale").set(0.0)
    solver = geo.createNode("vellumsolver", "solver")
    solver.setInput(0, seams, 0)
    solver.setInput(1, seams, 1)
    solver.setInput(2, body)
    solver.parm("substeps").set(2)
    return solver


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    panels, body = np.load(job["panels"]), np.load(job["body"])
    frames = body["frames"]
    cloth = panels["points"]
    seams = panels["seams"]
    hou.session.strata_drape = {
        "cloth": {
            "frames": cloth[None],
            "polygons": panels["triangles"].tolist(),
            "lines": seams.tolist(),
        },
        "body": {"frames": frames, "polygons": body["triangles"].tolist()},
    }
    hou.setFps(float(job["fps"]))
    hou.playbar.setFrameRange(1, len(frames))
    solver = _network()
    record_from = int(job["record_from"])
    out = np.zeros((len(frames) - record_from, len(cloth), 3), np.float32)
    for frame in range(1, len(frames) + 1):
        hou.setFrame(frame)
        geometry = solver.geometry()
        if geometry is None:
            errors = {n.name(): str(n.errors())[:400] for n in solver.parent().children() if n.errors()}
            raise RuntimeError(f"frame {frame}: {errors}")
        if frame % 50 == 0:
            print("frame", frame, "of", len(frames), flush=True)
        if frame - 1 >= record_from:
            points = np.frombuffer(geometry.pointFloatAttribValuesAsString("P"), np.float32).reshape(-1, 3)
            out[frame - 1 - record_from] = points[: len(cloth)]
    output = Path(job["output"])
    temporary = output.with_name(f".tmp-{output.name}")
    np.savez_compressed(temporary, points=out, triangles=panels["triangles"], panel=panels["panel"])
    temporary.replace(output)
    print("draped", output, out.shape, flush=True)


main()
