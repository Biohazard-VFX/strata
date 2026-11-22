# Strata dataset contract (v0.2.0)

The training side reads exactly this. Version string: `strata-dataset-0.2.0` (in every
`manifest.json` and `camera.json`). `schema.json` is the JSON Schema for both files;
`validate.py` checks a shot directory against it and against the image files.

0.2.0 changes the targets: layers keep the sky (float32, sky clusters with zero inverse depth),
recomposition frames carry raw-sample sidecars, and the flat geometry mask is the beauty
`foreground` part. 0.1.0 layer files (half, sky dropped) do not satisfy it.

## Layout

A dataset root holds one directory per shot; any directory with a `manifest.json` is a shot.

```
<shot_id>/
  manifest.json
  camera.json
  beauty/0001.exr ... 0024.exr
  deep/0001.exr ... 0024.exr     raw deep, kept on the frame stride (`manifest.render.deep_stride`)
  layers/0001.exr ... 0024.exr   K = 4 sky-inclusive layer targets for every frame
  samples/<shot_id>_NNNN/        raw-sample sidecars of every recomposition frame (the deep stride)
  qa/NNNN.json      raw QA acceptance of each frame (identity of every input it validated)
  scene/            optional: the USD the shot was rendered from (reproducibility, not training input)
  qa.json           optional: the generator's acceptance report
```

Entries whose names start with a dot (`.staging/`) belong to the writer; readers ignore them.

Frames are numbered from 1 and zero-padded to four digits, at 24 fps. `manifest.render.frames` is the
rendered range: most shots are 8-frame clips, a minority are 24-96-frame sequences.

## Units and conventions

- Metres everywhere. World is Y-up.
- Camera space: x right, y up, the camera looks down -Z (OpenGL/USD). `world_to_camera` is a
  4x4 row-major matrix for column vectors: `p_cam = M @ [x, y, z, 1]`.
- Pixel coordinates: origin at the top-left corner of the top-left pixel, u right, v down. A
  camera-space point projects to `u = cx + fx * x / -z`, `v = cy - fy * y / -z`.
- Z (flat and deep) is camera-plane depth: the distance along the optical axis, `-z_cam`,
  positive in front of the camera. Sky pixels hit the 3000 m backdrop (instance id 0).
- Colour: scene-linear ACEScg, not display-referred. Display and log transforms are applied
  at training time.
- Lens distortion: none in v0 (`distortion.model = "none"`).

## camera.json

`frames` has one record per rendered frame: `frame`, `width`, `height`, intrinsics `fx fy cx cy`
in pixels, `world_to_camera`, `distortion`, `shutter_open`/`shutter_close` in frames relative to
the frame (both 0 when motion blur is off), `fstop` (0 means depth of field off),
`focus_distance_m`, `focal_length_mm`, `sensor_width_mm`. `move` names the camera move
(`locked`, `pan_tilt`, `handheld`, `dolly`, `orbit`).

## manifest.json

`shot_id`, `seed`, `recipe_version`, `generator_versions` (module to version), `spec` (the fixed
choices the shot was built from), `scene` (sampled terrain, lighting, hero, camera summary),
`instance_ids` (named id ranges: `first_id`, `count`, and for scatter layers the prototype keys),
`assets` (every asset used: `id`, `source`, `license`, `origin`), `render` (renderer, resolution,
samples, frame range, AOV lists) and `timings` (filled after rendering: device, per-frame seconds,
VRAM).

## beauty/NNNN.exr

Multi-part scanline EXR, one part per AOV, all at the camera resolution:

| part | channels | meaning |
| --- | --- | --- |
| `C` | `R G B A` | raw path-traced beauty (half); the deep flattens to this |
| `albedo` | `albedo.R/G/B` | diffuse albedo |
| `N` | `N.x/y/z` | shading normal in camera space (half) |
| `roughness` | `roughness.roughness` | specular roughness of the front surface |
| `metallic` | `metallic.metallic` | metalness of the front surface |
| `Z` | `Z.Z` | camera-plane depth of the nearest surface (float, min filter) |
| `instance_id` | `instance_id.instance_id` | id of the nearest surface (float holding an integer) |
| `denoised` | `denoised.R/G/B/A` | `C` denoised by OIDN with albedo and normal guides: the model input; targets stay raw |
| `foreground` | `foreground.foreground` | geometry coverage of the pixel (float; 0 where only the sky is seen) |
| `coverage` | `coverage.coverage` | flat copy of the deep sample-alpha channel: 1 wherever any surface, sky included, is seen; not a foreground mask |

## layers/NNNN.exr

Scanline EXR with, for each layer k in 0..3 (front to back), float32 `L<k>.invdepth` (1/metres,
visible-coverage-weighted), `L<k>.A` (the layer's conditional alpha) and `L<k>.R/G/B`
(premultiplied colour composited within the layer). They are `strata_contract.layers.reduce_samples`
applied to every deep sample of the frame, sky included, with `sky` = (id == 0): samples of visible
coverage at most 1e-12 are dropped, the rest split at the largest three log-depth gaps above
ln(1.15) and merge front to back. Any layer holding visible sky keeps its alpha and colour and has
inverse depth 0. Sample alpha is the deep `coverage` channel (float), not the half `A`. Compositing
the four layers front to back reproduces the beauty `C`; an opaque sky-only pixel is one layer with
alpha 1, the sky colour and inverse depth 0. Training reads these for every frame.

## samples/<shot_id>_NNNN/

The raw samples of a recomposition frame (every frame on the deep stride from the first frame),
as uncompressed C-order NPY (`strata_contract.samples`). A frame directory is write-once: it appears
whole, by one rename, and is never changed afterwards; only an offline removal deletes it. It is a
real directory, never a symlink. `identity.json` records the SHA-256 of every file below, and its
own SHA-256 is the frame's identity (recorded by raw QA). `strata_contract.samples.read` opens the
directory once, reads every file through that handle and verifies each against `identity.json`:

| file | dtype, shape | meaning |
| --- | --- | --- |
| `noisy.npy` | float32 (3, H, W) | beauty `C` RGB |
| `denoised.npy` | float32 (3, H, W) | beauty `denoised` RGB |
| `deep/shape.npy` | int64 (2,) | (H, W) |
| `deep/offsets.npy` | int64 (H*W+1,) | row-major pixel sample offsets, first 0, last N |
| `deep/z.npy` | float32 (N,) | camera-plane depth, front to back within a pixel (stable for ties) |
| `deep/alpha.npy` | float32 (N,) | the sample's conditional alpha: deep `coverage` clipped to [0, 1] |
| `deep/rgb.npy` | float32 (N, 3) | premultiplied colour |
| `deep/ids.npy` | int64 (N,) | round(instance_id / coverage); sky samples are kept with id 0 |
| `identity.json` | JSON | `{"sha256": {"<file>": "<hex>"}}` for each file above |

## deep/NNNN.exr

Single-part deep scanline EXR with channels `R G B A` (half), `Z ZBack instance_id coverage`
(float), one sample list per pixel, samples unsorted. `R G B` are premultiplied by `A`.
`instance_id` and `coverage` (a constant 1 on every surface, sky included) are premultiplied by the
sample's alpha at float precision, so `coverage` is the sample's float alpha and the integer id of a
sample is `round(instance_id / coverage)`; do not divide by the half `A`, which is off by up to
id x 2^-11 and rounds alpha above 0.99976 to 1. Karma merges samples of the same surface that lie
within 1 mm in depth (deep compression 5), so a sample can stand for several pixel samples, and a
merge across two surfaces gives a non-integer id. ZBack - Z stays at most about 1 mm for surfaces.
Fully occluded surfaces are not stored. Beauty and deep share a one-pixel box filter: compositing
the samples front to back reproduces the beauty `C` part.

Every pixel ends in a sky sample: a camera-only backdrop sphere (radius 3000 m, the HDRI mapped as
Karma maps its dome, same exposure) with instance id 0, so compositing all samples front to back
reproduces the beauty `C` everywhere, partly covered pixels over sky included. Layer targets and
sidecars keep id-0 samples. The background sample is opaque,
finite (camera-plane Z up to 3000 m, below the 5000 m far clip; never inf) with Z == ZBack, and
`manifest.background` records its reserved id (0) and radius: compositors drop it with one
DeepExpression on instance_id; training masks depth losses on it (sky depth is undefined) and
supervises its colour and coverage.

Instance ids: 0 is the sky backdrop, 1 terrain, 2 hero skin, 3 hero groom (hair or fur),
1000 and up are scattered instances; `manifest.instance_ids` maps every id to its layer.
