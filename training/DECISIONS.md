# Training v0 decisions

- Isolate all training work on feat/training-v0 in the requested worktree; leave runtime and dataset contract ownership untouched.
- Start with K=4; sort samples by camera-plane Z, retain the largest K-1 consecutive log-depth gaps exceeding ln(1.15), break ties at the first sample, and merge each contiguous cluster with front-to-back transmittance.
- Store conditional cluster coverage and conditional premultiplied RGB, not global contribution; coverage-weighted inverse depths use visible sample weights, and invisible/empty slots are zero with no depth loss.
- Preserve scene-linear beauty and deep RGB in their original working space; the legacy v004 files do not prove ACEScg and are labeled unverified rather than silently retagged.
- Reuse the existing v004 sRGB conversion and fixed exposure for exact input comparability; pilot ACEScg inputs use an explicit AP1-to-linear-sRGB matrix and the same fixed display exposure, not a tone mapper.
- Preserve the 28/8 v004 train/holdout split and its existing mixed/fur masks for historical two-layer comparison; additionally evaluate native K-layer targets separately.
- Pin upstream DP2 commit aa389cbb0b74341664ed663a0966c859905cd6c0 in uv.lock; vendor no Apple source or weights into Strata.
- Mark absent legacy material/ID AOVs explicitly; reconstruct camera extrinsics from legacy look-at parameters, and label approximate focus rather than inventing exact optics.
