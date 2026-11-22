"""Build the dataset contract's packed training fixture through the actual producer."""

import json
import sys
from pathlib import Path

from fixtures import SKY, Sample, merged, write_shot

from strata_datagen.render.pack import pack


def main() -> None:
    shot = Path(sys.argv[1])
    pixels = [
        [merged(0.1, 0.5, {1: 0.5, 42: 0.5}, names={42: "hero_hair"}), SKY],
        [Sample(0.2, 0.25, 42, roles=(1, 1, 1)), SKY],
        [Sample(0.1, 0.5, 3, roles=(1, 0, 0)), SKY],
        SKY,
    ]
    write_shot(shot, [pixels, pixels], stride=2)
    path = shot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["instance_ids"] += [
        {"name": "thin", "first_id": 3, "count": 1},
        {"name": "hero_hair", "first_id": 42, "count": 1},
    ]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest))
    temporary.replace(path)
    pack(shot, range(1, 3))
    (shot / "deep/0002.exr").unlink()


if __name__ == "__main__":
    main()
