"""Report the immutable identity of the Blender (and Infinigen) running this script, for export caches.

Run: blender -b --factory-startup --python tool_identity.py -- JOB.json, or with Infinigen's Python
(bpy as a module): python tool_identity.py -- JOB.json
JOB.json: infinigen (also identify the importable infinigen package), output (JSON file to write).
Blender is its version, build hash and commit time, plus the digest of its binary when it is one;
Infinigen is its version and the digest of every file of its package, so an edited checkout differs.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import bpy


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _tree(root: Path) -> str:
    found = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    listing = {p.relative_to(root).as_posix(): _digest(p) for p in found}
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


def _blender() -> dict[str, str]:
    app = bpy.app
    identity = {
        "version": app.version_string,
        "build_hash": app.build_hash.decode(),
        "build_commit": f"{app.build_commit_date.decode()} {app.build_commit_time.decode()}",
        "python": _digest(Path(sys.executable)),
    }
    if app.binary_path:
        identity["binary"] = _digest(Path(app.binary_path))
    return identity


def _infinigen() -> dict[str, str]:
    infinigen = importlib.import_module("infinigen")
    return {"version": str(infinigen.__version__), "package": _tree(Path(infinigen.__file__).parent)}


def main() -> None:
    job = json.loads(Path(sys.argv[-1]).read_text())
    identity: dict[str, object] = {"blender": _blender()}
    if job["infinigen"]:
        identity["infinigen"] = _infinigen()
    Path(job["output"]).write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
