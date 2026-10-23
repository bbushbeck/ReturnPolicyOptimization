"""M3 prototype: minimum-cost subset meeting additive relevance threshold.

Prerequisites are enforced: if A requires B, selecting A requires B.
Exact exhaustive search is suitable only for small n (O(2^n * (n+d))).
This is NOT a measured quality guarantee for real LLM responses.
"""
from itertools import combinations
from .optimizer import Rule, cost, value, validate


def minimum_cost_quality(rules: list[Rule], min_relevance: int,
                         dependencies: dict[str, set[str]] | None = None) -> list[Rule] | None:
    validate(rules, sum(r.cost for r in rules))
    if min_relevance < 0:
        raise ValueError("min_relevance must be nonnegative")
    deps = dependencies or {}
    names = {r.name for r in rules}
    if not set(deps).issubset(names) or any(not ds.issubset(names) for ds in deps.values()):
        raise ValueError("Unknown dependency")
    best = None
    for size in range(len(rules) + 1):
        for subset in combinations(rules, size):
            selected = {r.name for r in subset}
            if any(k in selected and not required.issubset(selected) for k, required in deps.items()):
                continue
            if value(subset) < min_relevance:
                continue
            candidate = list(subset)
            if best is None or (cost(candidate), len(candidate), tuple(r.name for r in candidate)) < (cost(best), len(best), tuple(r.name for r in best)):
                best = candidate
    return best
