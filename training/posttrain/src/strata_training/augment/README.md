# Training-time augmentation

Per-sample, seeded augmentation for the data loader. Nothing here writes datasets; every call
draws fresh parameters from the caller's `numpy.random.Generator`.

## Hook

```python
from strata_training.augment.deep import Shot, load_samples
from strata_training.augment.pipeline import AugmentConfig, Donor, augment_sample, roles

sample = augment_sample(shot, shot_roles, targets, donor, rng, config)
# sample.image   (3, H, W) float32 scene-linear model input (degraded)
# sample.clean   (3, H, W) float32 clean recomposed input (composite-loss beauty)
# sample.layers  Layers with K slots rebuilt from the merged deep samples
# sample.record  JSON-able parameters that were drawn
```

`shot` is a `Shot(deep, noisy, denoised)`: deep samples, the noisy beauty `C` part and the
denoised beauty, all on the same window; `targets` are that window's cached K-layer targets
(`shot.deep.targets(K)`). They are returned or linearly regraded unless samples are added,
removed or graded per instance, in which case targets are rebuilt from the merged deep.
`roles(manifest)` reads hero and scatter id ranges.
`donor` is another `Donor(Shot, Roles)` cropped to the same window of a shot with the same
resolution, or `None`. `AugmentConfig.from_dict` reads a JSON object and rejects unknown keys.

Call the hook on the training crop (crop plus halo), not the full frame: deep merges and the
K-layer rebuild scale with the sample count. Decode each deep EXR once with `read_samples` and
`save_samples`; `load_samples` memory-maps the arrays and `Shot.crop` slices windows without
copying the rest of the frame.

Inputs change per sample, so encoder features cannot come from a cache built on clean beauties.

## Parts

- `noise.py`, `grain.py`: the studio ARRI Alexa grain group (Nuke Noise, film response curves,
  per-channel sharpen, ISO and exposure-offset scaling).
- `camera.py`: tone jitter, Rec.709 or LogC3 camera encode, unsharp mask, banding quantisation,
  4:2:0 JPEG, intra H.264/H.265 (PyAV), exact decode back to scene-linear ACEScg.
- `deep.py`: sorted deep samples with instance ids, crop/select/merge, visible contributions and
  `compose`, which rebuilds the input from denoised beauties for any merged deep.
- `occluders.py`: procedural blobs, stripes, grids, fences, wires and curves as deep samples.
- `pipeline.py`: configuration, sampling and the hook.

See `DECISIONS.md` for the rules these follow.
