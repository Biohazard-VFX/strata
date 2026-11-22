"""Fail if datagen or the dataset contract names a library asset or a studio project.

Identifiers are stored as SHA-256 digests of lowercase words so this public file never spells
them. Asset ids are allowed under strata_datagen/presets/ (fixed canary recipes name assets);
project codes are allowed nowhere. Add a word as the output of
python -c "import hashlib; print(hashlib.sha256(b'word').hexdigest())".
"""

import hashlib
import re
import sys
from pathlib import Path

ASSETS = frozenset(
    {
        "fa5bf879acb91bd614cfdbcaff2dc80acf09fcaa3a1cbc137deb406ce5978df9",
        "77ee1d93c48e3f3d510e751c91d5c0acfbabce5ccd0534b87a3014238b2debb6",
    }
)
PROJECTS = frozenset(
    {
        "0e903a02f06756b7c5a1d9fe3bb4fec2238fa62ea1b37268400264c9bf12564f",
        "ab5f71b12766214c76b3f15dae8e1943f020849b609d63cfd8497a2c25aabc8a",
        "f7800810c768fcb8d0d433f7661309c32d1f61758b10c7111c56e6e85cead019",
        "90c780c172554ae4138522189559f94ffaa2ac33c94985e8a8e0337846cc04fd",
        "2e5655c8539b17c8a404c0a4347bdc6919589f5497503ec54e209ae514f0b530",
        "c148bbdd2cbe65329b1924e386d0fe3f6249fe214ee641811b89a9e9fb373503",
        "7677a25c0ba2cda00b669b1f41ad86b1384fbe63ee7d0827854166503b54ab04",
        "bd98eaf40b601250114c1a7b9550700f26e5f524f1783207e47b6f0f3e8018a8",
        "c5a8fb3b29edf44effe4dc858261bce6ce8e6d5ccfb98b5c1d3039e8665bab99",
        "f4c07bd4be74c20308d919a973d4cec51f27ab0ef3a0acf97d8f949c91ef0cc6",
        "96da9625ce0e67a1060cae25b645a1ee54393c9670413c0a0956728a94252985",
        "a5ea709ca6045401d7c39460ef6c16b0524ceeb1133292ef49e81e462aa562c5",
        "33597e363a20e2e4f23d213cef40dbe970c6dfccc73004ce212dc37f0f115571",
        "a17b01209613daefcdb3fe2c2cc3a5d5e205ef4379c3f2b0afce1c413ef0a8a4",
        "9eb91108748ca6f0d935cec01275a14bb77c61b4b14c3f53b9b7532071fd2683",
        "16f0e0adb9a3ceac0681d97c2f00fe98bca90f7400a9cc67527261cfe6d6fef9",
        "16dd845a1a79e090d91ee31edc55923f69a718375a2851dffc68b8e0a7ad8182",
        "6625b07b6712fd640eb713503d8dd5c76144898f86ebe2e2335ac942a87a7e88",
        "5e687039b8bd769ad6271622bffc3e5c115ba2010e5970147a43e6c6a39dc616",
    }
)
SKIP_PARTS = {".venv", "__pycache__", ".ruff_cache", ".pytest_cache"}
WORD = re.compile(r"[A-Za-z0-9]+")


def _digests(text: str) -> set[str]:
    return {hashlib.sha256(w.lower().encode()).hexdigest() for w in WORD.findall(text)}


def _files(roots: list[Path]) -> list[Path]:
    return sorted(
        p
        for root in roots
        for p in root.rglob("*")
        if p.is_file() and not SKIP_PARTS & set(p.parts) and p.suffix not in {".lock", ".pyc"}
    )


def violations(roots: list[Path]) -> list[str]:
    found: list[str] = []
    for path in _files(roots):
        digests = _digests(path.as_posix() + "\n" + path.read_text(errors="ignore"))
        if digests & PROJECTS:
            found.append(f"{path}: names a studio project")
        if digests & ASSETS and "presets" not in path.parts:
            found.append(f"{path}: names a specific library asset outside presets/")
    return found


def main() -> None:
    found = violations([Path(arg) for arg in sys.argv[1:]])
    if found:
        sys.exit("\n".join(found))


if __name__ == "__main__":
    main()
