#!/usr/bin/env python3
"""Fetch official Linux x64 GPU SDK without checking binaries into git.

Prefer the upstream release. If unavailable, assemble the same C SDK from the
Microsoft NuGet headers and the official onnxruntime-gpu PyPI wheel (no install).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile

VERSION = "1.32.0"


def download(url, path, sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as source, path.open("wb") as out:
            shutil.copyfileobj(source, out)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if sha256 and digest != sha256:
        raise RuntimeError(f"checksum mismatch for {path}")
    return digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1]/"third_party/onnxruntime")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TMPDIR", str(Path.home()/".cache")))/"strata-downloads")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    root = args.dest.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"destination must be empty: {root}")
    name = f"onnxruntime-linux-x64{'' if args.cpu else '-gpu'}-{VERSION}"
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{name}.tgz"
    try:
        archive = args.cache/(name+".tgz")
        digest = download(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(args.cache, filter="data")
        shutil.copytree(args.cache/name, root, dirs_exist_ok=True)
        sources = [{"url":url,"sha256":digest}]
    except urllib.error.HTTPError as e:
        print(f"Release unavailable ({e.code}); using official NuGet/PyPI artifacts")
        package = "onnxruntime" if args.cpu else "onnxruntime-gpu"
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{VERSION}/json",timeout=30) as response:
            metadata = json.load(response)
        candidates = [f for f in metadata["urls"] if "manylinux" in f["filename"] and "x86_64" in f["filename"] and "cp312-" in f["filename"]]
        if len(candidates) != 1:
            raise RuntimeError("cannot identify the Linux x64 wheel")
        wheel = candidates[0]
        archive = args.cache/wheel["filename"]
        digest = download(wheel["url"], archive, wheel["digests"]["sha256"])
        (root/"lib").mkdir(parents=True,exist_ok=True)
        sources = [{"url":wheel["url"],"sha256":digest}]
        with zipfile.ZipFile(archive) as z:
            for entry in z.namelist():
                base = Path(entry).name
                if "/capi/" in entry and base.startswith("libonnxruntime") and ".so" in base:
                    (root/"lib"/base).write_bytes(z.read(entry))
        for suffix in ("", ".1"):
            (root/"lib"/("libonnxruntime.so"+suffix)).symlink_to(f"libonnxruntime.so.{VERSION}")
        nuget_url = f"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{VERSION}/microsoft.ml.onnxruntime.{VERSION}.nupkg"
        nuget = args.cache/f"ort-{VERSION}.nupkg"
        digest = download(nuget_url, nuget)
        sources.append({"url":nuget_url,"sha256":digest})
        (root/"include").mkdir(exist_ok=True)
        with zipfile.ZipFile(nuget) as z:
            for entry in z.namelist():
                if entry in ("LICENSE", "ThirdPartyNotices.txt"):
                    (root/entry).write_bytes(z.read(entry))
                if entry.startswith("build/native/include/") and entry.endswith(".h"):
                    (root/"include"/Path(entry).name).write_bytes(z.read(entry))
    (root/"sources.json").write_text(json.dumps(sources,indent=2)+"\n")
    print(f"SDK ready: cmake -S runtime -B build/normal -DONNXRUNTIME_ROOT={root}")


if __name__ == "__main__":
    main()
