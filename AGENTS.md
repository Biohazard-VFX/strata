# AGENTS.md

Rules for anyone, human or agent, changing Strata. They are not suggestions. When a review finds a new class of mistake, add a rule here and, where possible, a check in `tools/`.

## Architecture

- Directory is architecture. `runtime/` is C, `training/` and `datagen/` are Python, `contract/` is the only bridge between them. `hosts/` include only `runtime/include/strata.h`.
- Respect the layers enforced by `tools/check_all.sh`. Never weaken a check to make code pass; fix the code.
- One implementation of each rule. If logic exists, call it; do not reimplement it with different behaviour.
- No speculative code: no options, providers, backends or parameters without a caller and a test. Delete untested paths; re-add them with their test.
- The library never prints, exits or picks host policy. Report state through the API.
- No machine-specific tuning in shipped code (memory caps, thread counts, paths). Use library defaults or an explicit, documented, tested option.
- Public API comments state ownership, lifetime and thread-safety for every type and function that has them.

## Code

- C: Linux kernel style (`.clang-format`), C11, `-Werror`, sanitizer-clean. Explicit error handling; no macros that hide control flow.
- Python: ruff and pyright strict, both clean.
- Comments: none by default. One line only, for an external quirk or invariant the code cannot show. Never TODO, FIXME, or narration of what changed.
- No dead code, no defensive clutter, no fallbacks that hide failure.
- Write files atomically (temp file, then rename).

## Tests and gates

- A failing gate is a finding. Root-cause it. Never loosen a tolerance, skip a case or add a suppression to get green unless the cause is proven and the evidence is in the commit message.
- Gates must detect the failure they exist for: check signed bias, not only magnitudes; do not gate algebraic copies of the same value.
- Every correction from review becomes a test or a check, not prose.
- Accelerated contract ports must compare against the shared rule on producer-built fixtures and committed generic renderer crops in CI.
- Tests prove behaviour. A test that would pass with the code deleted is not a test.

## Decisions

`DECISIONS.md` files record architecture: one line per entry, at most 200 characters, stating the choice and, briefly, why.

- A decision constrains future code (formats, semantics, boundaries, algorithms, conventions).
- Not decisions, and never written there: caveats, incident logs, measurements, benchmark methodology, build steps, branch or worktree notes, coordination between agents, what one machine could or could not run. Results and measurements go to the results docs outside the repo; build steps go in the README.

## Repository

- No plans, research notes, scratch files, binaries or model weights in the repo.
- Symlinks in the repo are relative and stay inside it; links to machine storage live only at ignored paths.
- No client or project identifiers anywhere, and no client assets. Generators stay generic; specific assets live in external libraries referenced by id. Canary and preset recipes may name library asset ids.
- Conventional Commits, enforced by `tools/check_commits.sh` and the commit-msg hook. Commit small and often; one logical change per commit.
- No Fox Renderfarm tooling (SDK wrappers, submit, pricing or download scripts): it is proprietary and stays private. Datagen emits generic job lists. RunPod tooling is allowed. Never commit credentials.
- Work on a feature branch. `main` changes only by review.
