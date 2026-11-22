"""Fail unless DECISIONS.md is titled `# Decisions` and is one bullet per line of at most 200 characters."""

import sys
from pathlib import Path

LIMIT = 200
TITLE = "# Decisions"


def violations(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    found = [] if lines[:1] == [TITLE] else [f"{path}:1: must start with {TITLE!r}"]
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("# "):
            continue
        if not line.startswith("- "):
            found.append(f"{path}:{number}: not a one-line bullet")
        elif len(line) > LIMIT:
            found.append(f"{path}:{number}: {len(line)} characters, limit {LIMIT}")
    return found


def main() -> None:
    found = [v for arg in sys.argv[1:] for v in violations(Path(arg))]
    if found:
        sys.exit("\n".join(found))


if __name__ == "__main__":
    main()
