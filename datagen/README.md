# Strata datagen

Seeded synthetic shots for Strata: USD + MaterialX scenes rendered with Karma XPU into the dataset
contract in `../contract/dataset/` (beauty AOVs, deep, cameras, manifests).

## Layout

- `strata_datagen/generators/`: independent generators, seed and parameters in, USD prims out
  (terrain, scatter, materials, lighting, camera, hero, groom). A generator never imports another.
- `strata_datagen/scene/`: composes the generators into one shot from a `ShotSpec` (seed plus fixed
  choices) and writes `manifest.json` and `camera.json`.
- `strata_datagen/presets/`: fixed recipes (canaries) that name external-library assets.
- `strata_datagen/library/`: prepares a build root's `assets/` from external sources: Megascans
  (local sync, or fetched with nxc), HDRIs and an external hero library (`hero.json` + USD poses).
- `strata_datagen/render/`: render settings in USD and the husk driver (farm-ready job lists).
- `strata_datagen/qa/`: per-shot acceptance checks and the contact sheet.
- `blender/`: FBX to USD conversion, run inside Blender.

## Use

```sh
uv sync
strata-datagen prepare --root BUILD --sources sources.json      # assets/library.json
strata-datagen build --root BUILD --presets hero_in_terrain --first-seed 1 --count 10
strata-datagen render --root BUILD --husk /path/to/husk          # or run BUILD/jobs.sh on a farm
strata-datagen qa --root BUILD                                   # qa_report.json, contact_sheet.jpg
python ../contract/dataset/validate.py BUILD/shots/*
./check.sh                                                       # ruff, pyright, layer and identifier checks, tests
```

`sources.json` lists `megascans_roots`, `hdris`, `hero_library`, `heroes` and `max_variations`.
Run Python with `PYTHONPATH` unset when the environment exports another interpreter's path.
Design choices and measurements are in `DECISIONS.md`.
