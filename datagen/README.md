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
- `strata_datagen/render/`: render settings in USD and the husk driver. Farm job lists run
  `strata-datagen render-task`, which holds the shot lock, renders with a husk that inherits it,
  then packs, accepts and prunes under the same hold. A farm renders its own copy of a shot; its
  downloaded output folders enter a local shot only through `strata-datagen ingest`, which stages
  them in the shot and renames them in under the lock (sidecar frames stay write-once).
- `strata_datagen/qa/`: per-shot acceptance checks and the contact sheet.
- `strata_datagen/timeline.py`: smooth seeded curves (`timeline.Curve`, a cubic B-spline) for every
  time-varying parameter. A generator's `sample_*` returns one value per frame sample (frames plus
  a motion-blur handle each side, sample i at USD time first_frame - 1 + i); `author` writes them as
  time samples. QA bounds per-frame camera steps.
- `strata_datagen/render/pack.py`: after each render task, sky-inclusive K-layer targets for every
  frame (`strata_dataset.layers`, in `../contract/dataset`) and raw-sample sidecars on the deep
  stride. It never deletes: `accept` runs raw QA per frame and records it, then `prune` drops
  off-stride raw deep of accepted, unchanged frames only. Sidecars are write-once; to repack a
  changed frame, stop every reader and run `remove-samples` first. Every command that changes a shot (build, render,
  render-task, pack, accept, prune, remove-samples, qa) holds an exclusive `flock` on `<shot>/.lock`; a locked shot is
  refused, not waited for. The kernel releases the lock when the last process holding it exits.
- `strata_datagen/regions.py`: the one form of named skin regions for every hero (below).
- `blender/`: FBX to USD conversion, MPFB human export and creature export, run inside Blender.

## Ingest contract

`strata-datagen ingest` moves a download in unit by unit under the shot lock. A unit is one file
outside `samples/`, or one whole published sidecar frame (`samples/<frame>`), and lands by a single
rename, so a reader sees the old unit or the new one, never part of either. The ingest as a whole is
not atomic: if a rename fails, units already moved stay, the rest keep their old contents, staging is
removed and the lock is released. Re-running the same ingest completes it: identical published
frames are skipped and every other file is replaced.

## Use

```sh
uv sync
strata-datagen creatures --recipes creatures.json --library HEROES --infinigen-python PY  # creature heroes
strata-datagen prepare --root BUILD --sources sources.json      # assets/library.json
strata-datagen build --root BUILD --presets hero_in_terrain --first-seed 1 --count 10  # 720-frame shots
strata-datagen render --root BUILD --husk /path/to/husk          # or BUILD/jobs.sh: 60-frame render-task jobs
strata-datagen qa --root BUILD                                   # qa_report.json, contact_sheet.jpg
python ../contract/dataset/validate.py BUILD/shots/*
./check.sh                                                       # ruff, pyright, layer and identifier checks, tests
```

`creatures.json` lists creature recipes (Infinigen factory and seed, or a glTF model and animation,
poses, body length, coat classes, provenance); see `strata_datagen/library/creatures.py`. Install
Infinigen in its own Python 3.11 environment (`INFINIGEN_MINIMAL_INSTALL=True uv pip install -e
".[infinigen1]"`) and pass that interpreter; glTF animals use a system Blender.
`sources.json` lists `megascans_roots`, `hdris`, `hero_library`, `heroes` and `max_variations`.
A hero whose `hero.json` lists a `scalp` region gets a seeded hairstyle in every shot; see
`strata_datagen/library/heroes.py` for the manifest fields.
Run Python with `PYTHONPATH` unset when the environment exports another interpreter's path.
Design choices and measurements are in `DECISIONS.md`.

## Hero regions

Every hero (static pose, human clip, morph take, creature) marks named skin areas the same way: a
float **vertex** primvar `region_<name>` on the skin prim, one weight in [0, 1] per point, listed by
name in `hero.json` `regions`. Weights are soft: a library that labels faces writes each vertex's
share of labelled faces (`regions.from_faces`), so borders read as 0.5 and the weights serve as groom
density; a face belongs to a region when its corners' mean weight is at least 0.5
(`regions.face_mask`). `regions.read` rejects a missing primvar, a non-float or non-vertex primvar, a
wrong length or values outside [0, 1], and `prepare` reads every listed region of every hero.

| region | heroes | used by |
| --- | --- | --- |
| `scalp`, `brows`, `lashes`, `beard` | humans (no `lashes`), test heads | hairstyle generator; `scalp` turns it on, `beard` needs `facial_hair` |
| `face`, `ears`, `lips`, `body` | humans | analysis only |
| creature parts (`head`, `jaw`, `body`, `tail`, `leg`, ...) | creatures | coat layers (`library/coat.py`) |

`forward` (default `[0, 0, 1]`) is the facing direction in the pose files with Y up; `facial_hair`
(default false) allows stubble and beards. Hair grows per shot on the skin's rest shape (the clip's
`rest` primvar for animated heroes) and follows the animated skin: a similarity transform fitted to
the upper scalp per clip sample, plus each root's remaining offset at every hairstyle key.

## Human heroes

`strata-datagen humans --recipe recipe.json --library LIBRARY` builds animated humans into an external
hero library (recipe fields are documented in `strata_datagen/library/humans.py`). Per human it runs
MPFB in Blender (body, skin, eyes, CC0 clothes, `cmu_mb` rig), retargets one CMU clip per category
(walk, run, gesture, dance, turn) and writes `hero.json`, `usd/rest.usdc`, `usd/<clip>.usdc` and
`textures/`. Humans wear skinned MPFB clothing. List the keys under `heroes` in `sources.json` and
`prepare` copies them; shots pick a clip at least as long as the shot and place it on the terrain.

Animated `hero.json` (a hero with `clips` instead of `poses`):

| field | meaning |
| --- | --- |
| `key`, `kind` | library id; `kind` is `human` |
| `rest` | rest-pose USD (bind pose, MPFB parts only), default prim `/hero` |
| `clips.<name>` | `file`, `category`, `frames` (usable frames), `first_frame` (1: time code 0 and `frames + 1` are blur handles), `fps`, `subframes` (sample offsets, -0.25/0/0.25), `speed_mps`, `bounds_min`/`bounds_max` (body extent around the root, feet at y = 0), `track` (root position per time code), `source` |
| `skin_prim` | the skin mesh, `/hero/body`; topology is constant over time |
| `parts` | every mesh prim with its `kind` (`body`, `eyes`, `garment`) and source asset |
| `regions` | `scalp`, `face`, `brows`, `beard`, `ears`, `lips`, `body` (see Hero regions); scalp, face, ears and body partition the head and body, brows lie in the face, beard covers the lower face and throat |
| `forward`, `facial_hair` | `[0, 0, 1]`; beards allowed when every corner is male and past the teen ages |
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

Shots on a human with a scalp region grow a hairstyle on the `rest` points and carry it with the
animated skin (see Hero regions), across morph takes too.
