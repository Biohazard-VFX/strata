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

## Existing v004 smoke data

```sh
P=/mnt/work/life/projects/deep-depth
D=$P/data/strata_v0
uv run strata-train convert-v004 "$P" "$P/data/v004_contract"
uv run strata-train targets "$P/data/v004_contract" "$D/targets"
uv run strata-train freeze "$D/targets/index.json" ../eval/v004_split.json
TORCH_HOME=/mnt/work/hermes-scratch/torch-home uv run strata-train features "$D/targets/index.json" "$D/features_s" --preset depthpro2-s
TORCH_HOME=/mnt/work/hermes-scratch/torch-home uv run strata-train features "$D/targets/index.json" "$D/features_h_fp32" --preset depthpro2-h
uv run strata-train train configs/s_head.json
uv run strata-train train configs/s_decoder.json
uv run strata-train train configs/h_head.json
uv run strata-train train configs/h_decoder.json
uv run strata-train baselines "$D/targets/index.json" ../eval/v004_split.json "$D/features_h_fp32" "$D/baselines" "$P/results/v004_probe/probe_B.pt" "$P/results/v004_probe/normalization.npz"
uv run strata-train evaluate "$D/targets/index.json" ../eval/v004_split.json "$D/evaluation" --baselines "$D/baselines" --checkpoint "$D/runs/s_head/final.pt" --checkpoint "$D/runs/s_decoder/final.pt" --checkpoint "$D/runs/h_head/final.pt" --checkpoint "$D/runs/h_decoder/final.pt"
```

On this shared workstation every GPU command must additionally run under the standing exclusive `gpu.lock`, a Signet GPU-thread claim/release and `systemd-run --user -p MemoryMax=32G -p MemorySwapMax=0`. The examples above are commands, not permission to bypass that coordination. Cached prediction also supports `predict ... --device cpu` for a CUDA-disabled evaluation phase. Host resource policy is external: measure peak VRAM and require an additional 2 GiB free before acquiring a GPU phase. The library imposes no allocator or thread policy; the CLI's optional `--threads N` sets a positive Torch thread count explicitly. H+ encoder extraction streams original-fp32-weight blocks from CPU; the v004 adapter reuses the already verified dense H+ activations while building its encoder pyramid. New sequence extraction computes features with joint DP2 temporal attention. Frozen encoder parameters are not uploaded for cached decoder training.

For published native contract shots, all camera records belong to one shot-level split. If a manifest has no explicit split, a deterministic SHA256 ranking holds out max(1,floor(N/5)) shots; at least two completed shots are needed. Native ID 3 is the groom/fur region and IDs 2/3 define hero silhouettes. Legacy material and ID AOVs are not invented: its JSON bridge conforms to the published schema but explicitly reports image-AOV nonconformance. Check the pilot has all beauty/deep frames before preparing it.

## Representation and objective

Sort each pixel's deep point samples front to back and discard samples with visible contribution ≤1e−12 before clustering. Keep the largest K−1 consecutive log-depth gaps strictly above ln(1.15), with first-occurrence tie breaking. Merge each contiguous cluster with conditional front-to-back coverage and premultiplied linear RGB; use visible-coverage-weighted inverse depth. Empty or completely hidden slots are zero. Conditional cluster RGB and alpha reproduce the source sample composite without applying alpha twice. No target is resized or interpolated, including gradient pyramids: those use exact stride-1/2/4 pixel selections.

The deep head takes DP2's full-resolution `head[1]` features, scene-linear image and scalar inverse depth. A 48-channel stem plus six dilated residual convolutions predicts K×5 channels. Layer zero starts from the pretrained scalar head, with a positive 1e−3 inverse-depth floor where its ReLU returns zero; multiplicative log residuals refine it. Positive softplus log-depth increments put every later layer behind it. Sigmoid coverage and conditional premultiplied color form the other channels. Encoder frozen; compare head-only with trainable decoder and the first two original head operators. New head weights load with `strict=False`, and all non-head incompatibilities are rejected.

The objective is unweighted valid-layer log inverse-depth L1 plus three-scale first-order inverse-depth gradient L1, 4× soft-target coverage BCE-with-logits, 0.25× premultiplied-color L1 and 0.25× composite L1. Coverage/color/reconstruction emphasize mixed pixels by 12×; geometry is not confidence weighted. The coverage coefficient compensates averaging over four slots. There is no extra ordering or sparsity loss: depth ordering is architectural and empty coverage is explicitly supervised. BCE uses float32 logits, not a loss on rounded sigmoid probabilities, so opaque saturation cannot erase the correction gradient for mixed coverage. Historical checkpoints retain their original L1-trained weights.

Decoder configs use separate AdamW groups: the new head's `learning_rate` and the pretrained decoder's `decoder_learning_rate`, with linear `warmup_steps` then cosine decay. Encoder weights stay frozen. Histories include every loss term, pre-clipping gradient norms and LR per group, coverage MAE and maximum absolute alpha logit. Nonfinite loss or gradients fail immediately. `train configs/h_diagnostic.json` is the seeded 400-step diagnostic; set its ordinary data/output paths for the deployment. Resume rejects changes to objective, groups or schedule; L1 optimizer states cannot resume a BCE run.

## Frozen evaluation

The saved protocol preserves all 28 train and 8 pose-240 holdout frames. Training uses native crops and final fixed-step checkpoints, not best-holdout selection. `metrics.json` has pooled mixed-pixel alpha MAE, front/back and ordered inverse-depth median relative errors, layer color/count/order diagnostics, native and affine-aligned directed boundary F1, silhouette recall, calibration reliability bins, composite and fixed-magenta-plane insertion errors, and edge-band recomposition errors. Raw metric-depth AbsRel/δ1 are distinct from oracle global-scale-aligned diagnostics. The historical frozen probe remains oracle-scale-conditioned; it is not a raw metric competitor.

The original v004 reference table compares the first predicted coverage/depth and last GT-valid ordered depth slot with its two-layer GT; it never compacts away a transparent foreground or sorts predictions after inference. Native K-layer scores remain separate. Analytical methods use the same 95th/5th local percentiles and 20-bin curves fitted only on the frozen training IDs. Fake deep has a single opaque sample. The frozen probe can be rerun on new native data with its original weights and training-only normalization. Baseline color is observed beauty copied into both layers, so their zero self-composite error is tautological; the insertion test exposes geometry errors but does not establish hidden-color recovery.

Boundary evaluation is a documented paper-style adaptation: four directed adjacent-pixel contours; 5/10/15/20/25% thresholds weighted proportionally to threshold; exact native-pixel matching, with separate positive affine inverse-depth alignment. The papers do not fully specify the discrete threshold weights. Silhouette recall uses alpha>0.1 and scanline connected-run nonmaximum suppression; v004 lacks true instance masks, so its silhouette is explicitly a Z<3m coverage proxy, not a published segmentation benchmark. No label interpolation occurs in evaluation either.

Deep predictions retain every positive-coverage sample, including low-alpha HDR contributions. Outputs are NPZ plus point-sample deep EXRs (`R G B A Z ZBack`, ZBack=Z) for the requested first N holdout frames. Set the Nuke working color space to match the input EXR; legacy v004's scene-linear primaries are unverified and are not silently labeled ACEScg. No interactive Nuke canary acceptance is claimed.

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

`baselines ... --calibration PATH/curves.json` reuses the original training-only curves without fitting on OOD frames. An evaluation-only protocol can have an empty `train` list and all pilot frame IDs in `test`, with the exact index SHA256. Do not train on these shots. Native role masks decode `round(instance_id/A)` per deep sample, include any visible groom contribution, and threshold composited hero coverage at 0.1 for silhouettes.

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
invented fallbacks. The current datagen contract does not yet publish a denoised layer;
prepare supplied denoises explicitly until it does.

Build the clean cache from those same images:

```sh
strata-train features /data/targets/index.json /data/clean_features --preset depthpro2-h --samples /data/samples
```

Set `features` to that clean cache. Input-content hashes reject old noisy caches and
changed denoises. Raw deep samples remain available before rebuilding K-layer targets.
The exact training-side augmentation hook is `OnlineLoader.prepare` in
`strata_training.data.online`: it calls `augment.pipeline.augment_sample` on the native
crop plus halo, then transfers its image and matching targets. No augmented features
are written. `augmentation` passes explicit settings to `AugmentConfig.from_dict`;
use top-level `clean_fraction`, not `augmentation.clean`. Logs include loader seconds
and clean/live counts. CPU-heavy augmentation is not silently pre-baked.
