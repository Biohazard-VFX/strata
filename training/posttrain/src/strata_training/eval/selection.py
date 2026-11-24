from typing import Any

SWEEP_MIN_DELTA = 1e-4
LEAKAGE_MIN_DELTA = 1e-5


def selection_key(metrics: dict[str, Any]) -> tuple[bool, float, float, int]:
    leak = metrics["leakage"]
    score = float(metrics["coverage_sweep"]["selection"])
    leakage = float(leak["all"]["mean"])
    step = int(metrics["checkpoint_step"])
    return (False, score, leakage, step) if leak["passed"] else (True, leakage, score, step)


def validation_improves(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    return best is None or selection_key(candidate) < selection_key(best)


def meaningful_improvement(candidate: dict[str, Any], anchor: dict[str, Any] | None) -> bool:
    if anchor is None:
        return True
    if candidate["leakage"]["passed"] != anchor["leakage"]["passed"]:
        return bool(candidate["leakage"]["passed"])
    if candidate["leakage"]["passed"]:
        return (
            anchor["coverage_sweep"]["selection"] - candidate["coverage_sweep"]["selection"]
            >= SWEEP_MIN_DELTA
        )
    return (
        anchor["leakage"]["all"]["mean"] - candidate["leakage"]["all"]["mean"] >= LEAKAGE_MIN_DELTA
    )
