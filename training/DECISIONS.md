# Decisions

- CUDA training pins raw row spans and prepares four lookahead batches on a private thread/stream; events order consumption and record_stream protects consumer lifetimes.

- Torch transmission uses bounded row-local log-prefix scans, never a floating global CUDA scan, to make dense replay bitwise stable.

- CUDA hybrid batches default to Torch math; raw target rebuilds and Alexa fields batch across views, while final-image codecs alone use bounded CPU threads.

- Persistent spawned workers prepare native I/O; raw CSR samples cross queues as privately owned mmap descriptors, then gather crop columns on the device.
- Step-keyed RNG preserves worker/prefetch/resume identity; Torch uses independent stage flags and per-view math seeds (step-seed-gpu-v2), NumPy retains step-seed-v2.
- Shared K-layer reduction uses stable per-pixel compiled segments and float64 sums; sky membership is reduced in the same pass, not reclustered.
- Photometric-only crops never materialize raw samples; deep composition uses stable segmented merging and accumulation.
- Frame objects retain metadata only; bounded per-process LRU mappings serve immutable NPY targets, bases and features; compressed NPZ caches migrate atomically once.
- Deep crops concatenate only contiguous row sample spans from mmap offsets, never a whole-frame sample-index array.
- Deep targets use K=4 front-to-back slots at native resolution; labels are never resized or interpolated.
- Sort samples by Z and retain the largest K-1 log-depth gaps above ln(1.15); ties choose the first gap.
- Discard visible coverage <=1e-12 before clustering so fully occluded samples cannot consume slots.
- Store conditional coverage and premultiplied RGB; combine depths by visible-coverage-weighted inverse depth.
- Keep empty slots zero and exclude them from geometry losses.
- Preserve source scene-linear primaries; transform only the RGB input supplied to the DP2 encoder.
- Initialize front inverse depth from DP2; positive log-depth increments place later slots behind it.
- Freeze the DP2 encoder; train the deep head alone or the decoder and deep head.
- Geometry uses valid-layer log L1 and three native-stride first-order gradient scales, without confidence weighting.
- Coverage, color and compositing losses may emphasize partial coverage; geometry never uses predicted confidence.
- Read versioned camera frame lists and corner-origin pixel centers from the dataset contract.
- Layer heads act per frame on joint temporal features and accept DP2 streaming caches.
- Temporal feature contexts never cross shot, split or frame-gap boundaries.
- Cache frozen encoder pyramids so decoder training does not retain encoder activations.
- Omit frozen encoder tensors from trainable checkpoints and restore them from the pinned upstream model.
- Persist optimizer and RNG state in checkpoints so interrupted runs can continue.
- Reject checkpoint-key mismatches outside the declared new-head and frozen-encoder boundaries.
- Deep EXR export preserves every positive-coverage sample, including faint HDR contributions.
- Native role masks use stored linear foreground, hero and fur coverage; ids are best-effort per-object labels only.
- Post-training lives in training/posttrain so model export remains independently owned.
- Dataset ingestion accepts only strata_dataset.records.VERSION; old shots and role caches must be regenerated.
- OOD baselines reuse frozen training-only calibration; oracle-conditioned predictions retain checkpoint training-scale provenance.
- Supervise coverage with float32 soft-target BCE logits so sigmoid saturation cannot erase mixed-pixel correction gradients.
- Warm up new-head and pretrained-decoder optimizer groups at separate learning rates; reject resume across objective or schedule changes.
- Oracle evaluation aligns the current scalar prior before the deep head; frozen heads align the stock prior with the same GT mask.
- Recomposition consumes only producer-published samples through strata_dataset.samples.read, including its identity verification.
- NumPy clustering is owned by strata_dataset; the training-owned Torch adapter shares constants and must match its CPU/CUDA targets.
- Hybrid training uses native 512² independent crops, 40% clean-cache reuse and frozen bf16/SDPA encoding otherwise; augmented features are never cached.
- Sky membership comes from linear foreground coverage; sky keeps colour and alpha while clusters containing visible sky have no depth supervision.
- Select best checkpoints by OOD-validation mixed-alpha MAE every 250 steps; stop after four stale checks and treat step budgets as ceilings.
- Canary shots and real-matte sets are test-only: no training, early stopping or tuning; inherited manifest split=test-only blocks supervision, and validation uses held-out synthetic shots.
- Index sample eligibility routes off-stride frames through cached targets and photometric-only augmentation; donor selection uses eligible frames only.
- Target cache names include a digest of manifest, camera, packed beauty/layers, published sample identity, K and cache version; source changes never overwrite live mappings.
