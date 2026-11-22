# Strata training v0

Native-resolution K=4 deep supervision for Apple Depth Pro 2. The source and uv lock are self-contained; Apple weights download into `TORCH_HOME`. Weights and derivatives retain Apple's research license. This is a small synthetic smoke experiment, not a production-ready model.

## Install and checks

```sh
cd training/posttrain
env -u PYTHONPATH uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pyright
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --frozen pytest -q
```

The tests protect independently computed compositing/depth values, occlusion/empty samples, native three-scale gradients, mixed-pixel masking, directed boundary sign/location/scale behavior, calibration, temporal tracks, sequence isolation, model ordering and actual deep EXR read/write. Pyright is strict. `torch_api.py` isolates the small set of upstream Torch bindings whose annotations are incomplete; no strict diagnostics are disabled.

## Dataset migration

New ingestion accepts only `strata-dataset-0.4.0` through the shared records check.
The historical v004 converter is removed: older images lack linear role channels
and both identity checks, so relabelling their JSON cannot produce conformant data.
Re-render or drop old shots, and rebuild old role caches. Training configs, readers and evaluation labels must describe the current contract.

Run `bash tools/check_training.sh` from the checkout root for the isolated
post-training lint, strict types and CPU tests. It generates a tiny packed shot with
the real 0.4.0 producer and supplies it through `STRATA_DATASET_FIXTURE`; training
tests only consume the files through the dataset contract, never import datagen.
For direct pytest runs, generate that fixture with the datagen environment first
and set the variable to the packed shot directory. CUDA math tests require the
external GPU coordination and memory policy appropriate to the deployment.

## Representation and objective

Sort each pixel's deep point samples front to back and discard samples with visible contribution ≤1e−12 before clustering. Keep the largest K−1 consecutive log-depth gaps strictly above ln(1.15), with first-occurrence tie breaking. Merge each contiguous cluster with conditional front-to-back coverage and premultiplied linear RGB; use visible-coverage-weighted inverse depth. Empty or completely hidden slots are zero. Conditional cluster RGB and alpha reproduce the source sample composite without applying alpha twice. No target is resized or interpolated, including gradient pyramids: those use exact stride-1/2/4 pixel selections.

The deep head takes DP2's full-resolution `head[1]` features, scene-linear image and scalar inverse depth. A 48-channel stem plus six dilated residual convolutions predicts K×5 channels. Layer zero starts from the pretrained scalar head, with a positive 1e−3 inverse-depth floor where its ReLU returns zero; multiplicative log residuals refine it. Positive softplus log-depth increments put every later layer behind it. Sigmoid coverage and conditional premultiplied color form the other channels. Encoder frozen; compare head-only with trainable decoder and the first two original head operators. New head weights load with `strict=False`, and all non-head incompatibilities are rejected.

The objective is unweighted valid-layer log inverse-depth L1 plus three-scale first-order inverse-depth gradient L1, 4× soft-target coverage BCE-with-logits, 0.25× premultiplied-color L1 and 0.25× composite L1. Coverage/color/reconstruction emphasize mixed pixels by 12×; geometry is not confidence weighted. The coverage coefficient compensates averaging over four slots. There is no extra ordering or sparsity loss: depth ordering is architectural and empty coverage is explicitly supervised. BCE uses float32 logits, not a loss on rounded sigmoid probabilities, so opaque saturation cannot erase the correction gradient for mixed coverage. Historical checkpoints retain their original L1-trained weights.

Decoder configs use separate AdamW groups: the new head's `learning_rate` and the pretrained decoder's `decoder_learning_rate`, with linear `warmup_steps` then cosine decay. Encoder weights stay frozen. Histories include every loss term, pre-clipping gradient norms and LR per group, coverage MAE and maximum absolute alpha logit. Nonfinite loss or gradients fail immediately. `train configs/h_diagnostic.json` is the seeded 400-step diagnostic; set its ordinary data/output paths for the deployment. Resume rejects changes to objective, groups or schedule; L1 optimizer states cannot resume a BCE run.

## Frozen evaluation

The saved protocol preserves all 28 train and 8 pose-240 holdout frames. Training uses native crops. Historical v004 checkpoints used fixed final steps; new runs use the separate OOD-validation policy below. `metrics.json` has pooled mixed-pixel alpha MAE, front/back and ordered inverse-depth median relative errors, layer color/count/order diagnostics, native and affine-aligned directed boundary F1, silhouette recall, calibration reliability bins, composite and fixed-magenta-plane insertion errors, and edge-band recomposition errors. Raw metric-depth AbsRel/δ1 are distinct from oracle global-scale-aligned diagnostics. The historical frozen probe remains oracle-scale-conditioned; it is not a raw metric competitor.

The original v004 reference table compares the first predicted coverage/depth and last GT-valid ordered depth slot with its two-layer GT; it never compacts away a transparent foreground or sorts predictions after inference. Native K-layer scores remain separate. Analytical methods use the same 95th/5th local percentiles and 20-bin curves fitted only on the frozen training IDs. Fake deep has a single opaque sample. The frozen probe can be rerun on new native data with its original weights and training-only normalization. Baseline color is observed beauty copied into both layers, so their zero self-composite error is tautological; the insertion test exposes geometry errors but does not establish hidden-color recovery.

Boundary evaluation is a documented paper-style adaptation: four directed adjacent-pixel contours; 5/10/15/20/25% thresholds weighted proportionally to threshold; exact native-pixel matching, with separate positive affine inverse-depth alignment. The papers do not fully specify the discrete threshold weights. Silhouette recall uses alpha>0.1 and scanline connected-run nonmaximum suppression; v004 lacks true instance masks, so its silhouette is explicitly a Z<3m coverage proxy, not a published segmentation benchmark. No label interpolation occurs in evaluation either.

Deep predictions retain every positive-coverage sample, including low-alpha HDR contributions. Outputs are NPZ plus point-sample deep EXRs (`R G B A Z ZBack`, ZBack=Z) for the requested first N holdout frames. Set the Nuke working color space to match the input EXR; legacy v004's scene-linear primaries are unverified and are not silently labeled ACEScg. No interactive Nuke canary acceptance is claimed.

## OOD-validation early stopping

New runs default to `early_stopping: true`, `validation_every: 250` and
`validation_patience: 4`. Supply `validation_index`, `validation_protocol` and
`validation_features`; the protocol's `test` IDs are the validation set. Use a
separate scene-distribution cohort, not a different pose of a training scene. The
loader rejects the training index itself and overlapping shot/frame/target/manifest
identities. Reserve different shots again for the final OOD test; never select
checkpoints on that test set. Canary shots and real-matte evaluation datasets are
final tests only: never training, early stopping, checkpoint selection or tuning.
Mark their shot or dataset-root `manifest.json` with `"split": "test-only"`. The
marker is inherited through ancestor manifests and followed symlink targets; legacy
`eval_only: true` markers are also protected. Training and validation selection,
feature/sample loading and direct validation evaluation refuse these sources. Final
test scoring remains allowed. Use held-out **synthetic** shots for early stopping.

A step-zero validation establishes the initial best. Full native-frame validation
then runs every N steps and at the ceiling. Checkpoint selection uses pooled
first-layer mixed-pixel alpha MAE, not training loss. Four consecutive checks without
a lower MAE stop training. `best.pt` is the selected checkpoint; `last.pt` and
`final.pt` retain the actual stopping-step state. Pipeline evaluation uses `best.pt`.
`history.json` logs training loss and validation points, and `validation.json` retains
all validation loss/MAE points, best step and patience state. Validation loss uses
the same objective on full-frame predictions, with logits reconstructed from clipped
inference probabilities; selection uses unclipped predicted alpha MAE.

Resume retains validation history/patience and requires identical validation sources,
cadence and output directory. It rejects a checkpoint whose patience is exhausted.
`early_stopping: false` is explicit opt-out for historical reproduction and fixed
400-step diagnostics, not the default. The ordinary configs expect prepared OOD
validation assets under `/data/validation`; adjust these paths for the deployment.

## Sequences

One shot may contain multiple frame numbers. `features --clip-length N` encodes contiguous same-shot clips jointly with DP2's temporal attention; the deep head remains per-frame. `clip_length` in training selects matching-coordinate native crops from those clips. Shots cannot cross train/holdout boundaries, and gaps split clips. Use a separate feature directory for a different temporal context. The wrapper's ordinary forward also accepts native `(T,3,H,W)` clips and an optional DP2 `KVCache` for streaming. Published `camera.json` frame lists are read using fx/fy/cx/cy, with corner-origin pixel centers at +0.5.

When holdout sequences exist, evaluation computes front-layer depth, coverage and color residual jitter on native-pixel camera-reprojected tracks with a 2% GT-depth visibility gate, without per-frame alignment. This is explicitly a static-content proxy, not independently tracked dynamic deep samples. v004 consists of single-frame shots and therefore has no measured temporal result. Dynamic per-sample identity tracks remain a dataset requirement.

## RunPod recipe (not launched)

Mount a contract dataset at `/data/dataset` and persistent output/checkpoint storage at `/data`. Adjust the ordinary JSON config for dataset size, clip length and available GPU memory. The default uses S, frozen encoder, trainable decoder and four-frame temporal contexts (short/single-frame shots remain valid). Batch size counts clips. No paid resource is created by this repository.

```sh
docker build -f training/Dockerfile -t strata-training:v0 .
docker run --rm --gpus all --memory=32g --memory-swap=32g -v /absolute/persistent/data:/data strata-training:v0
```

The container's one-command entry point is `uv run --frozen --no-dev strata-train pipeline configs/runpod.json`: targets → frozen split → native/temporal features → training → evaluation/deep export. Use the same command in a uv-enabled RunPod image if building this Dockerfile there. Baseline assets are optional external research artifacts; run the `baselines` command with the H+ cache and frozen v004 weights to add all historical competitors on a new dataset. The pipeline's default report contains its trained model and matched scalar DP2 baseline, not fabricated unavailable probe results.

## Oracle and OOD prediction

`predict` accepts `--scale-mode oracle-aligned` to substitute the same GT-derived global-scale input used by the historical probe, and `--features PATH` to evaluate a checkpoint on a different feature cache. It records both training and inference configurations; checkpoint bytes remain unchanged. For a trainable decoder, a tiled scalar-only pass calibrates that checkpoint's current scalar prior using the same GT solid-surface rule as the probe cache, then a second pass conditions the nonlinear deep head on it. Reusing a stale stock-model scale after decoder training is not an oracle alignment. This is an oracle-conditioned diagnostic, not metric inference or oracle-trained retraining.

`baselines ... --calibration PATH/curves.json` reuses the original training-only curves without fitting on OOD frames. An evaluation-only protocol can have an empty `train` list and all pilot frame IDs in `test`, with the exact index SHA256. Do not train on these shots. Native roles read `roles.foreground`, `roles.hero` and `roles.fur` from every packed target; silhouettes threshold stored foreground coverage at 0.1.

See `DECISIONS.md` for design choices. Measured outcomes and diagnostics live outside the repository under the project's results/data directories.


## Hybrid native-crop training

New training configs default to `input_mode: "hybrid"`, `crop: 512`, `halo: 0`,
`clean_fraction: 0.4`. Clean samples reuse the clean denoised feature/pyramid cache;
changed samples run the frozen encoder live in bf16/no-grad with SDPA. Independent
images are batched without DP2 cross-frame mixing, preserving its single-frame prefix
semantics. The decoder/head still backpropagate normally. Native crops are not resized;
live scalar conditioning uses crop width / native focal length. Hybrid mode requires
raw scale and clip_length=1; explicit cached mode retains temporal cached training.
The historical v004 reproduction configs select `input_mode: "cached"` explicitly.

Supply `samples: "/data/samples"`. Each `<samples>/<frame-id>/` contains `deep/` in the
augmentation package's public memory-map format, `noisy.npy`, and `denoised.npy`
(scene-linear ACEScg CHW). `strata_training.data.online.prepare_sample(entry,
denoised_array, output_directory)` builds this boundary from the native contract and
an actual denoiser output. Missing denoised images/IDs are errors, never noisy or
invented fallbacks. Pass `None` as the denoised array to read the published `denoised`
beauty part directly. Feature extraction also uses native target-cache denoises by
default; it stores and hashes the exact model-input image for matching evaluation.
Raw C beauty stays separate as the reconstruction target.

Build the clean cache from those same images:

```sh
strata-train features /data/targets/index.json /data/clean_features --preset depthpro2-h --samples /data/samples
```

Set `features` to that clean cache. Input-content hashes reject old noisy caches and
changed denoises. Raw deep samples remain available before rebuilding K-layer targets.
The NumPy reference hook is `OnlineLoader.prepare` in
`strata_training.data.online`. CUDA hybrid training instead defers math from
`NativeBatches` to `LoadedBatch.to`, as described below. No augmented features are
written. `augmentation` passes explicit settings to `AugmentConfig.from_dict`;
use top-level `clean_fraction`, not `augmentation.clean`. Logs include loader seconds
and clean/live counts. `training.json` reports `recomposition_eligible_fraction`.
The index records published-sample eligibility before sampling: eligible frames may
recompose, occlude or grade objects; other frames retain cached targets and use
photometric/codec/grain transforms only. Donors are always eligible frames.
Target-cache filenames include their input-content digest (manifest semantics,
camera record, beauty, packed layers, sample identity, K and cache-code version).
Changed content builds a new cache without invalidating existing mmap readers.
Augmentation is not silently pre-baked.

### Parallel CPU preparation

Set `loader_workers` to a nonnegative integer (default 0) and `loader_prefetch` to a
positive number of batches per worker (default 2). With workers enabled, training
uses spawned, persistent PyTorch DataLoader workers, prefetches both augmentation
and clean feature/pyramid crops on CPU, pins completed batches for CUDA, and only
transfers tensors in the training process. No encoder/model object is sent to a
worker. Spawn avoids forking an initialized CUDA runtime. Choose the worker count
and BLAS thread policy explicitly for the deployment; the package does not impose
machine-specific thread counts. Peak image memory scales with workers × prefetch ×
batch/crop size and raw sample density, not frame count. Metadata still scales
linearly with the number of index records.

NumPy sampling is intentionally versioned as `step-seed-v2`: each absolute training step
uses `SeedSequence([seed, step])`. Crop selection and augmentation draws within a
step remain sequential. Version 2 routes by published-sample eligibility before
augmentation and draws donors only from eligible frames. This makes outputs independent of worker
count, queue scheduling and prefetch position, including resumed runs. It does
change the old single-global-RNG training trajectory; resume refuses checkpoints
without the matching sampling version rather than pretending to continue their
sequence. Calling `OnlineLoader.prepare` with the same selections and RNG retains
deterministic per-view semantics. `loader_seconds` now measures the main process's
wait plus device transfer; overlapped worker CPU time is not added to it.

### Device-side hybrid preparation

`loader_math: "torch"` is the default for CUDA hybrid training; CPU and cached-mode
runs default to `"numpy"`. The explicit NumPy path remains the reference. Spawned
workers read compact targets, denoised crops and clean cached inputs; changed views
carry raw CSR sample mappings plus value-only augmentation plans. Private COW NPY
mappings own their lifetimes and reopen by descriptor across queues, avoiding a raw
array copy into shared memory. Sources must remain immutable until the batch is
consumed. The parent pinning stage copies requested raw row spans as well as
image/compact/cache tensors into pinned host memory. Nonblocking device transfer
stages the requested native row span, then gathers the crop's columns on the device.
Wide frames therefore transfer extra samples outside the crop's horizontal extent,
but never samples outside its row span.

Training consumes `device_batches`, a context-managed, four-batch lookahead. A single
preparation thread runs nonblocking transfers and finalization on a private CUDA
stream while the consumer runs the encoder/decoder/backward/optimizer step. A ready
event orders consumption without synchronizing the whole device; `record_stream`
protects every returned tensor through consumer use. Exit joins outstanding work,
including early stopping and errors. CPU training remains synchronous. Lookahead
uses absolute-step seeds and does not advance checkpoint RNG state. GPU memory now
includes four prepared lookaheads plus the current train step; host pinning includes
raw row-span copies. `loader_seconds` measures consumer wait, not background work.

`LoadedBatch.to` runs deep selection, stable depth merge, transmission, per-object
and scene grading, all six procedural occluder families, photometric camera math
and Alexa grain on the selected device. Changed deep targets are rebuilt and
scattered together across the batch through
`strata_training.torch_layers.reduce_sorted_samples`;
that optional Torch module owns the same thresholds, tie breaking, hidden-sample
rule and sky masking as `strata_dataset.layers`. Reductions accumulate in float64.
Transmission uses row-local log-prefix scans sized by actual maximum pixel density,
avoiding CUDA global-scan carry-order variability without truncating samples.
Alexa noise fields are batched by profile, retaining the studio permutation table,
Nuke response curves and nearest-edge sharpening. Its tensor noise kernel compiles
lazily on CUDA; include first-use compilation in startup, not steady-state timing.

JPEG/H.264/H.265 encode/decode runs on CPU worker threads after GPU camera encoding,
on final cropped input images only. A batch owns at most one codec thread per
compressed image and joins before returning. Camera decoding and uncompressed
quantization run on the device. Targets never pass through grain, camera transforms
or codecs. Codecs are not pre-baked because their input depends on that view's
actual recomposition and grain.

The Torch sampling successor is `step-seed-gpu-v2`: retain the absolute-step seed,
then draw clean/live identity, independent augmentation-stage enable flags and a
per-view math seed. Donor selection follows that plan. This permits workers to omit
raw mappings for photometric-only views without executing augmentation. Worker
count, prefetch and resumed absolute-step slices preserve the same view. Resume
rejects changing between the two sampling versions. Direct NumPy `prepare` retains
its sequential per-view draw order.

Tests compare device and CPU/contract targets, composition, occluders and grain at
`atol=rtol=2e-6`; camera examples use `atol=5e-5, rtol=2e-5`. These are numerical,
not cross-device bitwise, equivalence gates. Codec quantization is discontinuous:
values at a bin boundary may select adjacent levels after floating-point rounding.
Same-device replays are checked bitwise. CPU tests can be isolated with
`CUDA_VISIBLE_DEVICES=`; the new math tests also exercise CUDA when visible and
must use the standing GPU resource protocol. Tests include batched fields/targets,
source immutability, exact native column gathering, halo targets, worker/resume
invariance and full deferred-batch comparison with independent CPU recomposition.

GPU memory scales with the live batch's actual raw sample count, including row-span
transfer and float64 scratch. Choose batch size, workers and prefetch explicitly;
there is no hidden density truncation or allocator policy. Native producer failures
below remain hard errors. Throughput measurements belong in the external results
directory, not this architecture/readme.

### Memory-mapped training caches

The first `Frame` construction migrates each compressed target/base NPZ to a sibling
`<name>.npz.mmap/` directory of individual NPY arrays, including exact row-major
mixed-pixel coordinates. Migration holds only one frame/archive at a time and publishes
the directory atomically; subsequent runs read only requested mmap windows. Allocate
additional disk space for the uncompressed cache. Caches are immutable: changing an
NPZ's size or modification time rejects its stale mapping directory; remove that
sibling directory before rebuilding. Frame objects hold metadata, not image arrays;
each worker retains at most two mapped archives, feature frames and raw-sample shots.
Raw deep crop reads gather contiguous sample spans for the requested pixel rows.

### Native datagen layout and supervision boundary

Only `strata-dataset-0.4.0` records are accepted, checked by
`strata_dataset.records.errors`. Target preparation reads sky-inclusive
`layers/NNNN.exr` and linear `roles.foreground`, `roles.hero`, `roles.fur` for every
frame, including frames without raw deep. Caches use `linear-roles-v3`; older role
caches must be regenerated. Roles retain fractional coverage without renormalising.

Recomposition reads `deep/roles.npy` alongside the other immutable sample sidecars.
Raw deep decoding calls `strata_dataset.identity.decode` with both checks and float
coverage. Id -1 means unlabelled: retain colour, alpha, depth and roles, but exclude
it from per-object selection and grading. Sky masking uses the foreground channel,
not an id or a depth threshold. Geometry and sky both retain colour/alpha targets;
only sky inverse depth is zero. The Torch reducer is training-owned and tested
against the dataset contract's NumPy reducer, including real packed frames.

## Target cache maintenance

`targets DATASET OUTPUT` memoizes source EXR digests in `OUTPUT/digests.json` using
size, nanosecond mtime and inode. Warm preparation reuses those digests, while
changed source metadata forces hashing again. Target archives remain content-addressed;
preparation never overwrites arrays held by an existing frame mapping.

Run `strata-train cache-prune OUTPUT` offline to remove target NPZ archives and NPY
mapping directories not referenced by `OUTPUT/index.json`. Both preparation and pruning
require an exclusive file lock. Training holds shared locks on its training and validation
target caches for the whole run, so offline maintenance refuses while either is in use.

Loader startup logs the fraction of producer-published, recomposition-eligible views.
`eligibility_warning_fraction` sets the warning threshold (default `0.25`, in `[0, 1]`).
Ineligible views use photometric-only inputs. Every clean, augmented and off-stride view
uses the composite of its final K-layer targets for beauty supervision, including cached
training and validation; rendered or denoised beauty remains an input, not a loss target.
