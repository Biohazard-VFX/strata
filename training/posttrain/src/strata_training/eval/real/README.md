# Real coverage evaluation

Run from `training/posttrain` with `env -u PYTHONPATH uv run --frozen python -m strata_training.eval.real`.
Data and results are external paths passed by the caller; nothing here is a training input.

## Protocol v1

Deep alpha is conditional, front to back. The primary diagnostic is fixed front slot
alpha[0]. This freezes the two-layer foreground/background convention used in synthetic
training, without selecting layers using test alpha, depth, trimaps or per-image fitting.
An explicit, independently predicted boolean foreground membership m can instead expose
visible target coverage: sum_k m_k alpha_k prod_{j<k}(1-alpha_j). Background samples still
occlude later target samples. Alpha-over of the whole scene is NOT a target matte: an
opaque background makes it one. Models with no target/instance assignment cannot solve
arbitrary foreground semantics; front-slot scores are an allocation/coverage proxy, not
a complete target-conditioned matting result. A 2D matte cannot validate deep geometry.

GT remains native resolution. Decode integer alpha using its original bit depth; never
resize labels, infer alpha from RGB, or silently clip invalid predictions. Mixed mask is
strictly 0<GT<1, with a separate 1/255 sensitivity score. Boundary is a five-pixel
Chebyshev dilation of both sides of 4-neighbour GT>=0.5 transitions and all mixed pixels.
Empty regions produce JSON null and are excluded, not zero, in dataset macro means.
MAE/MSE are unscaled [0,1] errors. SAD, Grad and Conn are full-frame sums /1000.
Report per-image values and unweighted image macro means, not pooled megapixel means.

Grad is the sigma=1.4, epsilon=.01, L2-normalized Gaussian derivative magnitude error,
with replicated borders. Conn is the technical-report variant, 4-connected largest
intersection component, thresholds .1 to 1, phi cutoff .15. Definitions track the MIT
RVM author's `evaluation/evaluate_lr.py` / `evaluate_hr.py` at commits
92d671f1d5d3d702795ed025f711d8c889e9c0ca / cb5d14669660bb1c5c3ef8c2be10bf4d9174eecf
(Git blob hashes; source: https://github.com/PeterL1n/RobustVideoMatting).
No prediction min/max renormalization, trimap restriction or fixed-resolution track is
silently applied, so do not compare these full-frame values to unknown-region tables.
Temporal error is RVM dtSSD =100 sqrt(mean(((p_t-p_prev)-(g_t-g_prev))^2)); score only
same-sequence, equal-resolution, adjacent integer frame indices, never a first-frame zero.
Sparse annotation intervals require a separate explicitly labelled cadence report.
Natural artist alpha, keyed/recomposited video, measured static controls, and relative
layer ordering are separate tracks; never aggregate their numbers.

## Acquisition

`fetch ROOT [--datasets aim500 am2k ...]` downloads evaluation-only resources and
records source URL, licence line, split, eval-only flag, byte count and SHA256 per file.
20 GiB per dataset, 80 GiB total and a 4 GiB free-space reserve are enforced before
transfer. Downloads publish atomically and declared size is enforced while streaming.
LayeredDepth SHA256 is checked against HF's upstream LFS digest. Other hashes are
first-acquisition integrity fingerprints, not proof of an author-provided checksum.
Public Google Drive listings paginate and enforce the official exact split count;
no 50-file partial-folder success. Existing data is reused only at the expected size.
Use a single fetch process per root. The manifests make blocked/manual requests explicit.
No login, signed agreement, non-commercial permission or mixed train/test archive is
bypassed. Dataset MIT terms do not clear third-party image/personality rights.
