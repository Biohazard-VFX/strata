# Strata dataset contract (v0.1.0)

The training side reads exactly this. Version string: `strata-dataset-0.1.0` (in every
`manifest.json` and `camera.json`). `schema.json` is the JSON Schema for both files;
`validate.py` checks a shot directory against it and against the image files.

## Layout

A dataset root holds one directory per shot; any directory with a `manifest.json` is a shot.

```
<shot_id>/
  manifest.json
  camera.json
  beauty/0001.exr ... 0024.exr
  deep/0001.exr ... 0024.exr     raw deep, kept on a frame stride (`manifest.render.deep_stride`)
  layers/0001.exr ... 0024.exr   K = 4 layer targets for every frame (`strata_contract.layers`)
  scene/            optional: the USD the shot was rendered from (reproducibility, not training input)
  qa.json           optional: the generator's acceptance report
```

Frames are numbered from 1 and zero-padded to four digits, at 24 fps. `manifest.render.frames` is the
rendered range: most shots are 8-frame clips, a minority are 24-96-frame sequences.

## Units and conventions

- Metres everywhere. World is Y-up.
- Camera space: x right, y up, the camera looks down -Z (OpenGL/USD). `world_to_camera` is a
  4x4 row-major matrix for column vectors: `p_cam = M @ [x, y, z, 1]`.
- Pixel coordinates: origin at the top-left corner of the top-left pixel, u right, v down. A
  camera-space point projects to `u = cx + fx * x / -z`, `v = cy - fy * y / -z`.
- Z (flat and deep) is camera-plane depth: the distance along the optical axis, `-z_cam`,
  positive in front of the camera. Pixels that see only the sky have flat `Z = 0` and no deep samples.
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
| `coverage` | `coverage.coverage` | geometry coverage of the pixel (float; 0 where only the sky is seen) |

## layers/NNNN.exr

Scanline EXR with, for each layer k in 0..3 (front to back), `L<k>.invdepth` (float, 1/metres,
coverage-weighted), `L<k>.A` (the layer's own alpha) and `L<k>.R/G/B` (premultiplied colour
composited within the layer), all half except invdepth. They are `strata_contract.layers.reduce_samples`
applied to the frame's deep samples: sort by Z, split at the largest three log-depth gaps above
ln(1.15), merge front to back. Training reads these for every frame; raw deep exists only on the
stride. Draw training frames sparsely along a shot for diversity and as full windows for
temporal training.

## deep/NNNN.exr

Single-part deep scanline EXR with channels `R G B A` (half), `Z ZBack instance_id coverage`
(float), one sample list per pixel, samples unsorted. `R G B` are premultiplied by `A`.
`instance_id` and `coverage` (a constant 1 per surface) are premultiplied by the sample's alpha at
float precision, so the integer id of a sample is `round(instance_id / coverage)`; do not divide by
the half `A`, which is off by up to id x 2^-11. Karma merges samples of the same surface that lie
within 1 mm in depth (deep compression 5), so a sample can stand for several pixel samples, and a
merge across two surfaces gives a non-integer id. ZBack - Z stays at most about 1 mm for surfaces.
Fully occluded surfaces are not stored. Beauty and deep share a one-pixel box filter: compositing
the samples front to back reproduces the beauty `C` part where the deep covers the pixel (mean
absolute difference about 2e-5); where the sky shows, the beauty adds it behind the deep.

Instance ids: 0 is sky (no deep sample), 1 terrain, 2 hero skin, 3 hero groom (hair or fur),
1000 and up are scattered instances; `manifest.instance_ids` maps every id to its layer.
