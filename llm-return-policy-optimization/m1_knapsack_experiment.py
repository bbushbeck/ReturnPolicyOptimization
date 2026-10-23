"""
M1 - Token Optimization for a Return Policy System
Preliminary synthetic experiment:
- Baseline: send all policy rules.
- Optimized: select rules using 0/1 Knapsack dynamic programming.

The rule costs and relevance values are illustrative, not measured from a real LLM.
"""

from time import perf_counter


def knapsack_01(rules, budget):
    """
    Solve 0/1 Knapsack using dynamic programming.
    Each rule is a dict with: name, tokens, relevance.
    Returns selected rules and algorithm execution time.
    """
    n = len(rules)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]

    start = perf_counter()

    for i in range(1, n + 1):
        cost = rules[i - 1]["tokens"]
        value = rules[i - 1]["relevance"]

        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]

            if cost <= b:
                candidate = dp[i - 1][b - cost] + value
                if candidate > dp[i][b]:
                    dp[i][b] = candidate

    # Reconstruct the selected subset.
    selected = []
    b = budget

    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            rule = rules[i - 1]
            selected.append(rule)
            b -= rule["tokens"]

    elapsed_seconds = perf_counter() - start
    selected.reverse()
    return selected, elapsed_seconds


def total_tokens(rules):
    return sum(rule["tokens"] for rule in rules)


def total_relevance(rules):
    return sum(rule["relevance"] for rule in rules)



def main():
    rules = [
        {"name": "Return window", "tokens": 120, "relevance": 9},
        {"name": "Receipt requirement", "tokens": 80, "relevance": 7},
        {"name": "Final-sale exception", "tokens": 100, "relevance": 10},
        {"name": "Damaged product", "tokens": 90, "relevance": 8},
        {"name": "Refund method", "tokens": 110, "relevance": 6},
        {"name": "Shipping conditions", "tokens": 130, "relevance": 4},
        {"name": "Warranty policy", "tokens": 140, "relevance": 3},
        {"name": "Exchange procedure", "tokens": 95, "relevance": 5},
    ]

    budget = 350

    # Baseline: send all rules.
    baseline_tokens = total_tokens(rules)
    baseline_relevance = total_relevance(rules)

    # Optimized method: select rules under the token budget.
    selected, selection_time = knapsack_01(rules, budget)
    optimized_tokens = total_tokens(selected)
    optimized_relevance = total_relevance(selected)

    reduction = (
        (baseline_tokens - optimized_tokens) / baseline_tokens * 100
        if baseline_tokens else 0.0
    )

    print("=" * 64)
    print("M1 PRELIMINARY EXPERIMENT - SYNTHETIC DATA")
    print("=" * 64)

    print(f"Number of policy fragments: {len(rules)}")
    print(f"Token budget: {budget}")
    print()

    print("--- BASELINE: ALL RULES ---")
    print(f"Total tokens: {baseline_tokens}")
    print(f"Total relevance: {baseline_relevance}")
    print()

    print("--- OPTIMIZED: KNAPSACK DP ---")
    print("Selected policy fragments:")

    for rule in selected:
        print(
            f"- {rule['name']} | "
            f"Tokens: {rule['tokens']} | "
            f"Relevance: {rule['relevance']}"
        )

    print()
    print(f"Selected fragments: {len(selected)}")
    print(f"Total tokens: {optimized_tokens}")
    print(f"Total relevance: {optimized_relevance}")
    print(f"Token reduction: {reduction:.2f}%")
    print(f"Algorithm execution time: {selection_time:.8f} seconds")
    print("=" * 64)

if __name__ == "__main__":
    main()
