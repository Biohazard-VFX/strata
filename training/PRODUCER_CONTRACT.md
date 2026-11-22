# Producer requirements for native hybrid supervision

This is a producer handoff, not permission for training to synthesize missing data.
The current compact packer drops sky, and the raw-deep stride omits identities on
most long-shot frames. Those datasets are **not qualified for hybrid training**.
Changing a manifest flag alone does not repair their contents. Existing
`compact_scene_only` entries remain rejected until the producer and consumer agree
on a versioned sky-inclusive contract and the produced files pass its validation.

## Every camera frame: sky-inclusive compact targets

Write `layers/NNNN.exr` on the original camera pixel grid, with K=4 front-to-back
slots. The required channels for each k=0..3 are:

| Channel | Required storage | Semantics |
| --- | --- | --- |
| `L<k>.invdepth` | float32 H×W | Visible-coverage-weighted inverse camera-plane depth, m⁻¹; zero for empty slots **and any cluster containing visible sky**. |
| `L<k>.A` | float32 H×W | The cluster's conditional coverage, not its already-transmitted visible coverage; preserve partial coverage. |
| `L<k>.R`, `.G`, `.B` | float32 H×W each | Conditional premultiplied scene-linear ACEScg color composited inside the cluster, including sky. No unpremultiplication, display transform or denoise. |

Use the shared `strata_contract.layers.reduce_samples` clustering rule, with actual
id-0 sky membership supplied through `sky`: discard visible sample coverage ≤1e−12,
split at the largest three consecutive log-depth gaps strictly above ln(1.15),
first occurrence winning a tie, and merge each cluster front to back. Do not remove
sky before clustering. Its alpha and RGB remain supervised even though geometry is
masked. If the K limit merges geometry and sky, the entire merged cluster's inverse
depth is zero. Empty slots are all zero. Target alpha must be in [0,1]; all stored
values must be finite. These float32 requirements avoid introducing another
coverage/color quantization at the producer boundary; old half-channel files are
not evidence of compliance.

An opaque sky-only pixel therefore has a nonzero color, alpha=1 and invdepth=0 in
one slot, not four zero slots. Partly covered foreground over sky must reconstruct
the raw `C` beauty by front-to-back compositing all slots. Validate this for sky-only,
edge/mixed, groom and ordinary geometry pixels. The flat beauty `coverage` AOV is
**geometry-only** and cannot substitute for sky coverage or per-cluster membership.
The exact new contract version/manifest declaration and its validator must be
published together with these files; no such version is inferred by training.

## Every frame eligible for recomposition: actual raw sample identities

A single nearest-object ID, the K compact slots, or identities borrowed from a
neighboring retained-deep frame are not sufficient. A compact cluster can contain
multiple instances. Preserve the renderer's actual per-sample depth, coverage,
premultiplied color and identity **before deleting stride-off raw EXRs**. Either
retain conformant raw deep EXR on every eligible frame, or publish the public mmap
boundary directly. The latter avoids decoding whole deep EXRs during training:

```text
<samples>/<frame-id>/
  noisy.npy
  denoised.npy
  deep/
    shape.npy
    offsets.npy
    z.npy
    alpha.npy
    rgb.npy
    ids.npy
```

`frame-id` is the literal target-index ID (native preparation uses
`<shot-id>_<four-digit-frame>`). Arrays are standalone, uncompressed, C-order NPY
files with `allow_pickle=False`; use little-endian fixed-width dtypes:

| File | dtype, shape | Meaning |
| --- | --- | --- |
| `shape.npy` | int64 `<i8`, (2,) | Native (H,W), matching camera, images and compact targets. |
| `offsets.npy` | int64 `<i8`, (H×W+1,) | Row-major pixel sample offsets; first=0, nondecreasing, last=N. Empty pixels repeat offsets. |
| `z.npy` | float32 `<f4`, (N,) | Finite, positive camera-plane depth in metres, stable front-to-back order within each pixel. Sky retains its actual finite backdrop depth here; only target geometry is zeroed. |
| `alpha.npy` | float32 `<f4`, (N,) | The sample's conditional A in [0,1], not global visible coverage. Promote source half A without inventing precision. |
| `rgb.npy` | float32 `<f4`, (N,3) | Sample RGB premultiplied by that A, scene-linear ACEScg. |
| `ids.npy` | int64 `<i8`, (N,) | Actual decoded instance identity; sky=0, terrain=1, hero skin=2, groom=3, scatter ranges as declared in the manifest. |
| `noisy.npy` | float32 `<f4`, (3,H,W) | Raw `C` beauty on the same pixel grid. |
| `denoised.npy` | float32 `<f4`, (3,H,W) | Actual denoiser output, finite scene-linear ACEScg, not noisy beauty copied as a fallback. |

For native deep EXR, decode IDs as
`round(float32(instance_id) / float32(coverage))`, **not** by dividing by half A.
Both source identity AOVs must be float channels. Preserve stable source order for
equal-depth samples. Keep id-0 sky samples and their true alpha/RGB. Do not invent
fully hidden surfaces absent from the render. The manifest must include actual
`instance_ids` role/range records, including names beginning `hero` and scatter
ranges carrying `prototypes`; identities must remain stable along a shot.

Publish each complete frame directory atomically. Keep raw/noisy/denoised images,
raw sample offsets, compact targets and camera records on the exact same grid.
Feature extraction must use the same denoised array bytes; its `image_sha256`
identity is checked by the loader. A raw-deep stride is still fine for archival
EXRs **only if** this genuine sample sidecar exists for every frame selected as a
hybrid background/donor. Missing data is a producer error, never a training-side
interpolation or identity fallback.
