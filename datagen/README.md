# Strata datagen

Seeded synthetic shots for Strata: USD + MaterialX scenes rendered with Karma XPU into the dataset
contract in `../contract/dataset/` (beauty AOVs, deep, cameras, manifests).

## Layout

- `strata_datagen/generators/`: independent generators, seed and parameters in, USD prims out
  (terrain, scatter, materials, lighting, camera, hero, groom, thin, hairstyle). A generator never
  imports another; shared strand operators (clump, curl, frizz, noise, braid) live in
  `strata_datagen/strands.py`.
- `strata_datagen/scene/`: composes the generators into one shot from a `ShotSpec` (seed plus fixed
  choices) and writes `manifest.json` and `camera.json`.
- `strata_datagen/presets/`: fixed recipes (canaries) that name external-library assets.
- `strata_datagen/library/`: prepares a build root's `assets/` from external sources: Megascans
  (local sync, or fetched with nxc), HDRIs and an external hero library (`hero.json` + USD poses).
- `strata_datagen/render/`: render settings in USD and the husk driver (farm-ready job lists).
- `strata_datagen/qa/`: per-shot acceptance checks and the contact sheet.
- `strata_datagen/timeline.py`: smooth seeded curves (`timeline.Curve`, a cubic B-spline) for every
  time-varying parameter. A generator's `sample_*` returns one value per frame sample (frames plus
  a motion-blur handle each side, sample i at USD time first_frame - 1 + i); `author` writes them as
  time samples. QA bounds per-frame camera steps.
- `strata_datagen/render/pack.py`: after each render task, K-layer targets for every frame
  (`strata_contract.layers`, in `../contract/dataset`) and raw deep kept on a frame stride.
- `blender/`: FBX to USD conversion, run inside Blender.
- `blender/`: FBX to USD conversion and MPFB human export, run inside Blender.
- `garments/`: GarmentCode pattern export, run in GarmentCode's Python; `houdini/`: Vellum drape, run in hython.

## Use

```sh
uv sync
strata-datagen creatures --recipes creatures.json --library HEROES --infinigen-python PY  # creature heroes
strata-datagen prepare --root BUILD --sources sources.json      # assets/library.json
strata-datagen build --root BUILD --presets hero_in_terrain --first-seed 1 --count 10  # 720-frame shots
strata-datagen render --root BUILD --husk /path/to/husk          # or BUILD/jobs.sh: 60-frame tasks, each husk + pack
strata-datagen qa --root BUILD                                   # qa_report.json, contact_sheet.jpg
python ../contract/dataset/validate.py BUILD/shots/*
./check.sh                                                       # ruff, pyright, layer and identifier checks, tests
```

`creatures.json` lists creature recipes (Infinigen factory and seed, or a glTF model and animation,
poses, body length, coat classes, provenance); see `strata_datagen/library/creatures.py`. Install
Infinigen in its own Python 3.11 environment (`INFINIGEN_MINIMAL_INSTALL=True uv pip install -e
".[infinigen1]"`) and pass that interpreter; glTF animals use a system Blender.
`sources.json` lists `megascans_roots`, `hdris`, `hero_library`, `heroes` and `max_variations`.
A hero whose `hero.json` lists a `scalp` region (float vertex primvar `region_scalp` on its skin) gets a
seeded hairstyle in every shot; see `strata_datagen/library/heroes.py` for the manifest fields.
Run Python with `PYTHONPATH` unset when the environment exports another interpreter's path.
Design choices and measurements are in `DECISIONS.md`.

## Human heroes

`strata-datagen humans --recipe recipe.json --library LIBRARY` builds animated humans into an external
hero library (recipe fields are documented in `strata_datagen/library/humans.py`). Per human it runs
MPFB in Blender (body, skin, eyes, CC0 clothes, `cmu_mb` rig), retargets one CMU clip per category
(walk, run, gesture, dance, turn) and writes `hero.json`, `usd/rest.usdc`, `usd/<clip>.usdc` and
`textures/`. Humans in the recipe's `garmentcode` range wear a GarmentCode garment sewn and draped in
Houdini Vellum instead of MPFB tops and bottoms. List the keys under `heroes` in `sources.json` and
`prepare` copies them; shots pick a clip at least as long as the shot and place it on the terrain.

Animated `hero.json` (a hero with `clips` instead of `poses`):

| field | meaning |
| --- | --- |
| `key`, `kind` | library id; `kind` is `human` |
| `rest` | rest-pose USD (bind pose, MPFB parts only), default prim `/hero` |
| `clips.<name>` | `file`, `category`, `frames` (usable frames), `first_frame` (1: time code 0 and `frames + 1` are blur handles), `fps`, `subframes` (sample offsets, -0.25/0/0.25), `speed_mps`, `bounds_min`/`bounds_max` (body extent around the root, feet at y = 0), `track` (root position per time code), `source` |
| `skin_prim` | the skin mesh, `/hero/body`; topology is constant over time |
| `parts` | every mesh prim with its `kind` (`body`, `eyes`, `garment`, `garment_sim`) and source asset |
| `regions` | `scalp`, `face`, `brows`, `beard`, `ears`, `lips`, `body`: GeomSubset prims (face indices) on the skin, family `region_family` (`hero_region`); scalp, face, ears and body partition the head and body, brows lie in the face, beard covers the lower face and throat |
| `rest_primvar` | `rest`: bind-pose points on every part, for binding grooms in rest space and following the animated points |
| `phenotype`, `face_targets`, `skin`, `eyes`, `outfit` | MPFB recipe of the human |
| `source`, `license`, `licenses`, `redistributable` | provenance; `licenses` has one line per source |

`strata-datagen morphs --recipe recipe.json --library LIBRARY` builds morphing humans (recipe block
`morph`): each family is a few MPFB corner humans sharing one outfit, so all parts keep one topology.
A take (default 30 s) blends the corners' rest points, rigs and skin textures with seeded smooth
weights (`library/curves.py`) while CMU clips chained with crossfades (`library/motion_chain.py`)
play, so gender, age, body shape, height, skin tone and face drift continuously with no cuts or
pops; clothes follow because they are skinned and blended like the body, and root travel scales with
the blended hip height so feet stay grounded. Morph heroes have `kind` `human_morph`, `corners`
(each corner's skin, phenotype and face targets) and per clip `parameters`: a JSON file with the
corner weights, the blended phenotype per frame and the chained CMU windows. The skin look mixes the
corner skins through `albedo_mix<i>` nodes whose `mix` inputs are time-sampled. Regions are on the
fixed topology, so they move with the morph.

A groom generator binds strands to skin faces in a region using `rest`, then moves each root with its
face's animated points at every time sample of the clip.
