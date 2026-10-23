"""Pure selectors for illustrative integer rule costs and additive relevance.

No tokenizer, LLM, or real policy is integrated. Tie-breaking is deterministic.
"""
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class Rule:
    name: str
    cost: int
    relevance: int


def validate(rules: list[Rule], budget: int) -> None:
    if budget < 0 or any(r.cost <= 0 or r.relevance < 0 for r in rules):
        raise ValueError("Budget must be nonnegative; costs positive and scores nonnegative")
    if len({r.name for r in rules}) != len(rules):
        raise ValueError("Rule names must be unique")


def knapsack(rules: list[Rule], budget: int) -> list[Rule]:
    """Exact 0/1 dynamic programming, O(nB) time and space."""
    validate(rules, budget)
    dp = [[0] * (budget + 1) for _ in range(len(rules) + 1)]
    for i, rule in enumerate(rules, 1):
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if rule.cost <= b:
                dp[i][b] = max(dp[i][b], dp[i - 1][b - rule.cost] + rule.relevance)
    chosen = []
    b = budget
    for i in range(len(rules), 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            chosen.append(rules[i - 1]); b -= rules[i - 1].cost
    return chosen[::-1]


def greedy_ratio(rules: list[Rule], budget: int) -> list[Rule]:
    """Descending relevance/cost, stable deterministic ties; not always optimal."""
    validate(rules, budget)
    ranked = sorted(rules, key=lambda r: (-r.relevance / r.cost, -r.relevance, r.name))
    chosen, remaining = [], budget
    for rule in ranked:
        if rule.cost <= remaining:
            chosen.append(rule); remaining -= rule.cost
    return chosen


def top_relevance(rules: list[Rule], budget: int) -> list[Rule]:
    """Highest relevance first; skip rules that do not fit."""
    validate(rules, budget)
    ranked = sorted(rules, key=lambda r: (-r.relevance, r.name))
    chosen, remaining = [], budget
    for rule in ranked:
        if rule.cost <= remaining:
            chosen.append(rule); remaining -= rule.cost
    return chosen


def baseline(rules: list[Rule], budget: int | None = None) -> list[Rule]:
    """Unoptimized full-manual baseline; intentionally NOT budget constrained."""
    return list(rules)


def cost(rules: Iterable[Rule]) -> int:
    return sum(r.cost for r in rules)


def value(rules: Iterable[Rule]) -> int:
    return sum(r.relevance for r in rules)
