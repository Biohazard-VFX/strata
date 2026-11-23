# Native hybrid supervision

The authoritative producer format is `contract/dataset/README.md` and
`strata_dataset.records.VERSION`: only `strata-dataset-0.5.0` is accepted.
Re-render or drop older datasets; changing their version strings does not add
linear roles or recover missing identity checks.

Read sky-inclusive K=4 targets and `roles.foreground`, `roles.hero`, `roles.fur`
from every packed frame, even after raw deep is pruned. Role coverage is fractional,
not binary and not renormalised. The flat `coverage` AOV includes sky and is not a
foreground mask. Sky membership comes only from the linear foreground role.

Each recomposition frame needs the producer's immutable `samples/<frame-id>/`
sidecar, including `deep/roles.npy`, or conformant raw deep with both identity
checks and linear roles. The reader verifies sidecar identities through
`strata_dataset.samples.read`. Native alpha comes from float `coverage`, never
half `A`. Raw identity decoding calls `strata_dataset.identity.decode`.

Id -1 is unlabelled, not sky: keep its colour, alpha, geometry and roles; omit it
only from per-object selection/grading. Role supervision never groups by object
id. Per-object augmentation may group only labelled samples. Every transported,
cropped, selected, merged or graded sample carries its original premultiplied
roles. Procedural occluders explicitly author foreground coverage.

Compact-only frames may train in cached mode, but hybrid backgrounds and donors
need actual raw samples on the same native pixel grid. No interpolation, invented
per-sample ids, or neighbouring-frame role masks are allowed. Feature extraction
uses the matching denoised bytes; the loader checks `image_sha256`.
