import json
from pathlib import Path
from typing import Any, Literal

TEST_ONLY = "test-only"


def is_test_only(path: Path) -> bool:
    """Read manifest policy for a borrowed path and its ancestors, following both alias and target.

    No files are modified; calls share no mutable state. Policy files must be valid JSON objects.
    """
    seen: set[Path] = set()
    for source in (path.absolute(), path.resolve()):
        directory = source.parent if source.is_file() else source
        candidates = ([source] if source.is_file() else []) + [
            parent / "manifest.json" for parent in (directory, *directory.parents)
        ]
        for manifest in candidates:
            if manifest in seen or not manifest.is_file():
                continue
            seen.add(manifest)
            data: dict[str, Any] = json.loads(manifest.read_text())
            if data.get("split") == TEST_ONLY or data.get("eval_only") is True:
                return True
    return False


def require_supervision(
    entries: list[dict[str, Any]],
    *,
    purpose: Literal["training", "validation"],
    root: Path | None = None,
) -> None:
    """Reject protected supervision sources; borrow entries without mutation."""
    if root is not None and is_test_only(root):
        raise ValueError(f"{purpose} refuses test-only dataset root: {root}")
    for entry in entries:
        paths = [Path(entry[key]) for key in ("manifest", "dataset_root") if entry.get(key)]
        if entry.get("cache"):
            paths.append(Path(entry["cache"]).parent)
        if (
            entry.get("split") == TEST_ONLY
            or entry.get("test_only") is True
            or entry.get("eval_only") is True
            or any(is_test_only(path) for path in paths)
        ):
            raise ValueError(f"{purpose} refuses test-only source: {entry['id']}")
