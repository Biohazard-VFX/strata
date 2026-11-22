"""The locked environment provides every module the export pipeline imports."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]

type Table = dict[str, object]


def locked_project() -> Table:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    packages = cast("list[Table]", lock["package"])
    return next(package for package in packages if package["name"] == "strata-train")


def extra_requirements(extra: str) -> dict[str, str]:
    metadata = cast("Table", locked_project()["metadata"])
    requirements = cast("list[Table]", metadata["requires-dist"])
    return {
        cast("str", requirement["name"]): cast("str", requirement.get("specifier", ""))
        for requirement in requirements
        if requirement.get("marker") == f"extra == '{extra}'"
    }


class Dependencies(unittest.TestCase):
    def test_cuda_extra_pins_onnxruntime_gpu(self) -> None:
        self.assertEqual(extra_requirements("cuda").get("onnxruntime-gpu"), "==1.32.0")

    def test_cuda_extra_resolves_onnxruntime_gpu(self) -> None:
        dependencies = cast("Table", locked_project()["optional-dependencies"])
        cuda = cast("list[Table]", dependencies["cuda"])
        self.assertIn("onnxruntime-gpu", [package["name"] for package in cuda])


if __name__ == "__main__":
    unittest.main()
