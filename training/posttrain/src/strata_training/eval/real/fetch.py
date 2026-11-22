import hashlib
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from strata_training.io import atomic_path, write_json

GIB = 1024**3
CATALOG: dict[str, dict[str, Any]] = {
    "aim500": {
        "source": "https://github.com/JizhiziLi/AIM",
        "licence": "MIT dataset agreement; source-image rights retained",
        "split": "test",
        "expected": 500,
        "folders": {
            "image": "12YYRDmJVzUjDywkx4gSdGxOdNcfGTGV6",
            "alpha": "16D6c77sS4GL9THEfdcpgxkBV83cPFDjb",
        },
    },
    "am2k": {
        "source": "https://github.com/JizhiziLi/GFM",
        "licence": "MIT dataset agreement; source-image rights retained",
        "split": "validation (official 200-image test)",
        "expected": 200,
        "folders": {
            "image": "1f_cixl1iRmtUYD3ml6Jx_P4-CJQ6EjO8",
            "alpha": "1CWwLqhkYg8JdeD-bxxL35Y0-DlCKuVFh",
        },
    },
    "p3m500p": {
        "source": "https://github.com/JizhiziLi/P3M",
        "licence": "MIT dataset agreement; image/personality rights retained",
        "split": "P3M-500-P",
        "expected": 500,
        "mirror": "https://huggingface.co/datasets/nobg/P3M-10K",
        "hf_repo": "nobg/P3M-10K",
        "hf_prefix": "data/P3M_500_P-",
    },
    "videomatte240k": {
        "source": "https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md",
        "licence": (
            "MIT confirmed by BackgroundMattingV2 issue 208; "
            "backgrounds have separate source rights"
        ),
        "split": "published RVM composited HD evaluation; not natural capture",
        "url": "https://robustvideomatting.blob.core.windows.net/eval/videomatte_1920x1080.tar",
    },
    "vm108": {
        "source": "https://github.com/yunkezhang/TCVOM",
        "licence": "Academic/non-commercial only",
        "split": "28 validation clips",
        "blocked": (
            "Owner must clear corporate evaluation rights and request validation-only export; "
            "full archive is 192 GB."
        ),
    },
    "alphamatting": {
        "source": "https://static.alphamatting.com/datasets.html",
        "licence": "Dataset commercial evaluation permission unverified",
        "split": "27 public labels frozen as evaluation control",
        "blocked": (
            "Owner must obtain commercial evaluation permission; "
            "hidden test alpha cannot be fetched."
        ),
    },
    "crgnn": {
        "source": "https://github.com/TiantianWang/VideoMatting-CRGNN",
        "licence": "Dataset grant unverified; code/model research-only",
        "split": "19 real videos / sparse annotated frames",
        "blocked": (
            "Owner must request separate real-dataset rights; "
            "sparse labels are not dense temporal GT."
        ),
    },
    "layereddepth": {
        "source": "https://huggingface.co/datasets/princeton-vl/LayeredDepth",
        "licence": "CC0-1.0 real benchmark",
        "split": "validation",
        "expected": 300,
    },
}


def request(url: str) -> Any:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Strata-eval/1"}), timeout=120
    )


def drive_files(folder: str) -> list[dict[str, Any]]:
    with request("https://drive.google.com/drive/folders/" + folder) as response:
        html = response.read().decode()
    for key in sorted(set(re.findall(r"AIza[\w-]+", html))):
        files: list[dict[str, Any]] = []
        token = ""
        try:
            while True:
                query = urllib.parse.urlencode(
                    {
                        "key": key,
                        "q": f"'{folder}' in parents and trashed=false",
                        "pageSize": 1000,
                        "fields": "files(id,name,size),nextPageToken",
                        "pageToken": token,
                    }
                )
                with request("https://www.googleapis.com/drive/v3/files?" + query) as response:
                    page = json.load(response)
                files.extend(page["files"])
                token = page.get("nextPageToken", "")
                if not token:
                    return files
        except urllib.error.HTTPError as error:
            if error.code not in (400, 403):
                raise
    raise RuntimeError("Public Drive listing unavailable; no partial-folder success")


def checksum(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(url: str, path: Path, size: int, sha256: str | None = None) -> dict[str, Any]:
    if size <= 0 or size > 20 * GIB:
        raise ValueError("Unknown or oversized download")
    if not path.exists():
        with request(url) as response, atomic_path(path) as temporary:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise ValueError("Download returned HTML, possibly an agreement/login wall")
            with temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("Download exceeded declared size")
                    output.write(chunk)
            if total != size:
                raise ValueError("Truncated download")
            if sha256 is not None and checksum(temporary) != sha256:
                raise ValueError("Upstream SHA256 mismatch")
    if path.stat().st_size != size:
        raise ValueError("Existing file size mismatch")
    digest = checksum(path)
    if sha256 is not None and digest != sha256:
        raise ValueError("Existing file SHA256 mismatch")
    return {
        "path": str(path.name),
        "bytes": size,
        "sha256": digest,
        "source_url": url,
        "upstream_sha256": sha256,
    }


def fetch(root: Path, names: list[str], workers: int = 8) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for name in names:
        metadata = CATALOG[name]
        target = root / name
        target.mkdir(exist_ok=True)
        previous_path = target / "manifest.json"
        previous: dict[str, Any] = (
            json.loads(previous_path.read_text()) if previous_path.exists() else {}
        )
        known = {f["path"]: f for f in previous.get("files", [])}
        manifest: dict[str, Any] = {**metadata, "eval_only": True, "files": [], "status": "pending"}
        try:
            if "blocked" in metadata:
                manifest["status"] = "manual_request"
                summary[name] = manifest
                write_json(target / "manifest.json", manifest)
                continue
            jobs: list[tuple[str, Path, int, str | None]] = []
            if "folders" in metadata:
                for role, folder in metadata["folders"].items():
                    files = drive_files(folder)
                    if len(files) != metadata["expected"]:
                        raise ValueError(f"Incomplete {role} listing: {len(files)}")
                    for item in files:
                        filename = item["name"]
                        if Path(filename).name != filename:
                            raise ValueError("Unsafe Drive filename")
                        jobs.append(
                            (
                                "https://drive.google.com/uc?export=download&id=" + item["id"],
                                target / role / filename,
                                int(item["size"]),
                                None,
                            )
                        )
            elif name == "layereddepth" or "hf_repo" in metadata:
                repo = metadata.get("hf_repo", "princeton-vl/LayeredDepth")
                prefix = metadata.get("hf_prefix", "data/validation-")
                with request(
                    f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
                ) as response:
                    entries = json.load(response)
                for item in entries:
                    if item["path"].startswith(prefix):
                        jobs.append(
                            (
                                f"https://huggingface.co/datasets/{repo}/resolve/main/"
                                + item["path"],
                                target / Path(item["path"]).name,
                                item["size"],
                                item["lfs"]["oid"],
                            )
                        )
                if len(jobs) != (9 if name == "layereddepth" else 1):
                    raise ValueError("Unexpected evaluation shard count")
            else:
                with request(metadata["url"]) as response:
                    size = int(response.headers["Content-Length"])
                jobs.append((metadata["url"], target / "eval.tar", size, None))
            needed = sum(size for _, path, size, _ in jobs if not path.exists())
            existing = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            if sum(j[2] for j in jobs) > 20 * GIB or needed + existing > 80 * GIB:
                raise ValueError("Dataset/total byte budget exceeded")
            if shutil.disk_usage(root).free < needed + 4 * GIB:
                raise ValueError("Insufficient free disk plus 4 GiB reserve")

            def run(
                job: tuple[str, Path, int, str | None],
                directory: Path = target,
                fingerprints: dict[str, Any] = known,
            ) -> dict[str, Any]:
                relative = str(job[1].relative_to(directory))
                prior = fingerprints.get(relative)
                if (
                    job[1].exists()
                    and prior is not None
                    and prior["source_url"] == job[0]
                    and checksum(job[1]) != prior["sha256"]
                ):
                    raise ValueError("Existing file differs from acquisition fingerprint")
                result = download(*job)
                result["path"] = relative
                return result

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(run, jobs):
                    manifest["files"].append(result)
                    write_json(target / "manifest.json", manifest)
            manifest["status"] = "downloaded"
            manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
        except (OSError, ValueError, RuntimeError) as error:
            manifest["status"] = "blocked"
            manifest["error"] = str(error)
            manifest["files"] = list(
                {**known, **{f["path"]: f for f in manifest["files"]}}.values()
            )
        write_json(target / "manifest.json", manifest)
        summary[name] = manifest
    write_json(root / "fetch-summary.json", summary)
    return summary
