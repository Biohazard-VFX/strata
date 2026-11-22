"""Export GarmentCode sewing patterns as flat 3D panels with seam vertex pairs, one .npz per job.

Run with a Python that has GarmentCode (MIT) importable, from its repository root:
  python garmentcode_panels.py jobs.json
jobs.json: [{"body": measurements (YAML or JSON), "design": design.yaml, "name": str,
"output": panels.npz, "resolution_cm": float}]. Output arrays: points (metres, Y up, panels placed
around the body as in GarmentCode's box mesh), uv (flat panel coordinates, metres), triangles,
panel (index per point) and seams (pairs of point indices that GarmentCode stitches together).
Only the pattern code is used; GarmentCode's simulator is not.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from assets.bodies.body_params import BodyParameters
from assets.garment_programs.meta_garment import MetaGarment
from pygarment.meshgen import boxmeshgen
from pygarment.meshgen.boxmeshgen import BoxMesh

CM = 0.01
# GarmentCode's warning for two-vertex edges reads an attribute Edge never sets.
boxmeshgen.Edge.name = "edge"


def _pattern(job: dict[str, object], folder: Path) -> Path:
    body = BodyParameters(str(job["body"]))
    design = yaml.safe_load(Path(str(job["design"])).read_text())["design"]
    garment = MetaGarment(str(job["name"]), body, design)
    pattern = garment.assembly()
    if garment.is_self_intersecting():
        raise RuntimeError(f"{job['name']}: self-intersecting panels")
    out = Path(
        pattern.serialize(str(folder), to_subfolder=True, with_3d=False, with_text=False, view_ids=False)
    )
    return next(out.glob("*_specification.json"))


def _panels(spec: Path, resolution: float) -> dict[str, np.ndarray]:
    box = BoxMesh(str(spec), resolution)
    box.load()
    points: list[np.ndarray] = []
    flat: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    panel_of: list[np.ndarray] = []
    shared: dict[int, list[int]] = {}
    offset = 0
    for index, name in enumerate(box.panelNames):
        panel = box.panels[name]
        local = np.asarray(panel.rot_trans_panel(panel.panel_vertices), dtype=np.float64)
        faces = np.asarray(panel.panel_faces, dtype=np.int64)
        for loc in range(panel.n_stitches):
            shared.setdefault(box.verts_loc_glob[(name, loc)], []).append(offset + loc)
        points.append(local)
        flat.append(np.asarray(panel.panel_vertices, dtype=np.float64)[:, :2])
        triangles.append(faces + offset)
        panel_of.append(np.full(len(local), index))
        offset += len(local)
    seams = [(group[0], other) for group in shared.values() for other in group[1:]]
    return {
        "points": np.concatenate(points) * CM,
        "triangles": np.concatenate(triangles),
        "panel": np.concatenate(panel_of),
        "uv": np.concatenate(flat) * CM,
        "seams": np.asarray(seams, dtype=np.int64).reshape(-1, 2),
    }


def main() -> None:
    jobs = json.loads(Path(sys.argv[1]).read_text())
    for job in jobs:
        with tempfile.TemporaryDirectory() as tmp:
            arrays = _panels(_pattern(job, Path(tmp)), float(job["resolution_cm"]))
        output = Path(job["output"])
        temporary = output.with_name(f".tmp-{output.name}")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
        print("panels", output, len(arrays["points"]), len(arrays["seams"]), flush=True)


main()
