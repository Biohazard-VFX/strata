from typing import Any


def selection_key(metrics: dict[str, Any]) -> tuple[bool, float, float, int]:
    leak = metrics["leakage"]
    score = float(metrics["coverage_sweep"]["selection"])
    leakage = float(leak["all"]["mean"])
    step = int(metrics["checkpoint_step"])
    return (False, score, leakage, step) if leak["passed"] else (True, leakage, score, step)


def validation_improves(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    return best is None or selection_key(candidate) < selection_key(best)
