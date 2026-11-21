# Training v0 decisions

- Isolate all training work on feat/training-v0 in the requested worktree; leave runtime and dataset contract ownership untouched.
- Start with K=4; sort samples by camera-plane Z, retain the largest K-1 consecutive log-depth gaps exceeding ln(1.15), break ties at the first sample, and merge each contiguous cluster with front-to-back transmittance.
- Store conditional cluster coverage and conditional premultiplied RGB, not global contribution; coverage-weighted inverse depths use visible sample weights, and invisible/empty slots are zero with no depth loss.
- Preserve scene-linear beauty and deep RGB in their original working space; the legacy v004 files do not prove ACEScg and are labeled unverified rather than silently retagged.
- Reuse the existing v004 sRGB conversion and fixed exposure for exact input comparability; pilot ACEScg inputs use an explicit AP1-to-linear-sRGB matrix and the same fixed display exposure, not a tone mapper.
- Preserve the 28/8 v004 train/holdout split and its existing mixed/fur masks for historical two-layer comparison; additionally evaluate native K-layer targets separately.
- Pin upstream DP2 commit aa389cbb0b74341664ed663a0966c859905cd6c0 in uv.lock; vendor no Apple source or weights into Strata.
- Mark absent legacy material/ID AOVs explicitly; reconstruct camera extrinsics from legacy look-at parameters, and label approximate focus rather than inventing exact optics.
- Adopt paper_notes.md: never resize/interpolate depth or deep labels; compute the three first-order inverse-depth gradient scales with native stride-1/2/4 selection, not a six-scale or Laplacian loss.
- Do not confidence-weight geometry; mixed-pixel weighting is confined to coverage, color and reconstruction, with 4× coverage loss compensating four-slot averaging.
- Train primary models on raw focal-converted metric inverse depth and report oracle-scale-aligned diagnostics separately; the historical probe remains explicitly oracle-conditioned.
- Clamp the pretrained scalar ReLU output at 1e-3 inverse metres before multiplicative residual refinement so zero stock predictions can learn finite surfaces; the first S trial exposed 645,932 zero pixels across 36 frames.
- Keep the historical reference comparison separate from native K-layer metrics; collapse predictions at the largest visible log gap using a fixed 0.003 contribution cutoff without accessing GT.
- Use directed four-neighbor boundaries at 5/10/15/20/25% with threshold-proportional weights and exact native matching; disclose this discrete adaptation rather than claiming exact published-benchmark reproduction.
- Report v004 silhouette recall only as a Z<3m coverage proxy because object-ID masks are absent; use alpha>0.1 and connected-run nonmaximum suppression.
- Report observed-beauty baseline RGB as an explicit fallback, never as recovered hidden colors; supplement tautological self-composites with a fixed Z=3m magenta insertion test.
- Stream H+ frozen encoder blocks from CPU in bf16 to honor 12 GiB free VRAM; reuse existing v004 H+ dense activations, and save newly reassembled pyramids as fp16.
- Keep the entire frozen encoder on CPU during cached decoder training and omit it from new checkpoints; reload its pinned original weights before restoration.
- Split temporal contexts at shot, split and frame-gap boundaries; the per-frame deep head supports joint DP2 multi-frame encoder features, without treating unrelated photos as a temporal clip.
- Compute sequence jitter on visibility-gated static reprojection tracks without per-frame alignment when sequences exist; disclose that dynamic per-sample identity tracks are not supplied by v004.
- Discard zero-coverage/fully hidden samples before choosing cluster gaps so invisible depth gaps cannot consume visible layer slots; a new failing toy test exposed this and all 36 v004 caches remain bit-identical after the repair.
- Freeze final step budgets before holdout evaluation: S head 5,000, S decoder 1,500, H+ head 40,000, H+ decoder 1,500; evaluate final checkpoints only.
- Preserve the canonical Conventional Commit validator through a temporary hook bridge because the shared repository installed a hook pointing to tools absent from this older worktree; do not disable validation.

