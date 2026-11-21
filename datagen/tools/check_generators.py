"""Fail if a generator imports another generator: generators are composed only by scene/."""

import ast
import sys
from pathlib import Path

PACKAGE = "strata_datagen.generators"


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names.extend(f"{base}.{alias.name}" for alias in node.names)
            names.append(base)
    return names


def violations(folder: Path) -> list[str]:
    found: list[str] = []
    generators = {p.stem for p in folder.glob("*.py") if p.stem != "__init__"}
    for path in sorted(folder.glob("*.py")):
        for name in _imports(ast.parse(path.read_text())):
            relative = name.startswith(".")
            target = name.lstrip(".").split(".")
            absolute = name.startswith(PACKAGE + ".") and name.split(".")[2] in generators
            if (relative and target[0] in generators) or absolute:
                found.append(f"{path}: imports generator {name}")
    return found


def main() -> None:
    found = violations(Path(sys.argv[1]))
    if found:
        sys.exit("\n".join(found))


if __name__ == "__main__":
    main()
