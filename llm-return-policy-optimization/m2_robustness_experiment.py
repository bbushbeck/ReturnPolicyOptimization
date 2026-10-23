"""M2 - Robustness and Deterministic Evaluation (synthetic, no LLM).

Extends the original M1 0/1 Knapsack dynamic programming algorithm.
Run: python3 m2_robustness_experiment.py
Only Python standard library is required.

IMPORTANT: Accuracy here is a deterministic rule-based decision proxy, NOT
accuracy of an LLM. .
"""

from collections import Counter
from statistics import mean, pstdev
from time import perf_counter
import csv
import json
from pathlib import Path


# The original M1 algorithm, preserved (including tie behavior).
def knapsack_01(rules, budget):
    """Solve 0/1 Knapsack with dynamic programming and reconstruct selected rules."""
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


# Same 8 rule names and illustrative costs as M1.
# Their textual meanings below are NEW synthetic assumptions for M2.
RULES = [
    {"name": "Return window", "tokens": 120, "relevance": 9},
    {"name": "Receipt requirement", "tokens": 80, "relevance": 7},
    {"name": "Final-sale exception", "tokens": 100, "relevance": 10},
    {"name": "Damaged product", "tokens": 90, "relevance": 8},
    {"name": "Refund method", "tokens": 110, "relevance": 6},
    {"name": "Shipping conditions", "tokens": 130, "relevance": 4},
    {"name": "Warranty policy", "tokens": 140, "relevance": 3},
    {"name": "Exchange procedure", "tokens": 95, "relevance": 5},
]

BUDGET = 350
REPETITIONS = 30
# Synthetic, explicit scoring: relevant rules score 10; irrelevant rules score 1.
# This is a controlled test fixture, NOT semantic retrieval by an LLM.
RELEVANT_SCORE = 10
IRRELEVANT_SCORE = 1

# Gold decisions and necessary evidence sets are constructed for this synthetic
# policy. Multiple rules are sometimes necessary for a complete decision.
CASES = [
    {"id": "C01", "query": "Return unused item after 10 days with receipt", "gold": "APPROVE", "required": ["Return window", "Receipt requirement"]},
    {"id": "C02", "query": "Return unused item after 45 days with receipt", "gold": "REJECT", "required": ["Return window"]},
    {"id": "C03", "query": "Return final-sale item with no defect", "gold": "REJECT", "required": ["Final-sale exception"]},
    {"id": "C04", "query": "Return damaged final-sale item", "gold": "REVIEW", "required": ["Final-sale exception", "Damaged product"]},
    {"id": "C05", "query": "Return item without a receipt", "gold": "REVIEW", "required": ["Receipt requirement"]},
    {"id": "C06", "query": "Ask for original-payment refund on eligible return", "gold": "APPROVE", "required": ["Return window", "Refund method"]},
    {"id": "C07", "query": "Return item with damaged shipping packaging", "gold": "REVIEW", "required": ["Shipping conditions", "Damaged product"]},
    {"id": "C08", "query": "Request repair of defective product under warranty", "gold": "REVIEW", "required": ["Warranty policy", "Damaged product"]},
    {"id": "C09", "query": "Exchange unused item within return window", "gold": "APPROVE", "required": ["Exchange procedure", "Return window"]},
    {"id": "C10", "query": "Exchange final-sale item that arrived damaged", "gold": "REVIEW", "required": ["Exchange procedure", "Final-sale exception", "Damaged product"]},
]


def case_rules(case):
    """Copy M1 items; change relevance only for this fixed query."""
    necessary = set(case["required"])
    return [{**rule, "relevance": RELEVANT_SCORE if rule["name"] in necessary else IRRELEVANT_SCORE}
            for rule in RULES]


def evaluate_selection(case, selected):
    """Gold-evidence coverage and a conservative deterministic decision proxy.

    If all gold-required rules are present, the synthetic oracle returns the
    reference label; otherwise it abstains (INSUFFICIENT_CONTEXT). This is not
    a simulated LLM response and must not be reported as LLM accuracy.
    """
    names = {rule["name"] for rule in selected}
    required = set(case["required"])
    completeness = len(names & required) / len(required)
    prediction = case["gold"] if required <= names else "INSUFFICIENT_CONTEXT"
    return {
        "prediction": prediction,
        "correct": int(prediction == case["gold"]),
        "completeness": completeness,
        "all_required_present": int(required <= names),
    }


def run_experiment():
    if len({r["name"] for r in RULES}) != len(RULES):
        raise ValueError("Duplicate rule names")
    valid_names = {r["name"] for r in RULES}
    for case in CASES:
        if not case["required"] or not set(case["required"]) <= valid_names:
            raise ValueError("Invalid required rule in " + case["id"])
    rows = []
    times = []
    baseline_predictions = []
    optimized_predictions = []
    for case in CASES:
        rules = case_rules(case)
        baseline = evaluate_selection(case, rules)
        baseline_predictions.append(baseline["prediction"])
        baseline_cost = total_tokens(rules)
        results = []
        for repetition in range(1, REPETITIONS + 1):
            selected, elapsed = knapsack_01(rules, BUDGET)
            names = tuple(rule["name"] for rule in selected)
            cost = total_tokens(selected)
            assert cost <= BUDGET
            result = evaluate_selection(case, selected)
            results.append((names, result["prediction"], cost, result["completeness"]))
            times.append(elapsed)
        # For deterministic fixed inputs, the exact same selected subset and
        # decision should be produced on all repetitions.
        assert len(set(results)) == 1, "Non-deterministic result for " + case["id"]
        names, prediction, cost, completeness = results[0]
        optimized_predictions.append(prediction)
        rows.append({
            "case_id": case["id"], "query": case["query"],
            "reference_answer": case["gold"],
            "required_rules": "; ".join(case["required"]),
            "selected_rules": "; ".join(names),
            "baseline_tokens": baseline_cost,
            "optimized_tokens": cost,
            "reduction_percent": round(100 * (baseline_cost - cost) / baseline_cost, 4),
            "baseline_accuracy_proxy": baseline["correct"],
            "optimized_accuracy_proxy": int(prediction == case["gold"]),
            "baseline_completeness": baseline["completeness"],
            "optimized_completeness": completeness,
            "optimized_prediction": prediction,
            "selection_consistency": 1.0,
            "decision_consistency": 1.0,
        })
    baseline_total = sum(r["baseline_tokens"] for r in rows)
    optimized_total = sum(r["optimized_tokens"] for r in rows)
    summary = {
        "experiment": "M2 deterministic synthetic policy evaluation",
        "model": None,
        "model_version": None,
        "temperature": None,
        "sampling_parameters": "Not applicable: no LLM is called",
        "knapsack_budget": BUDGET,
        "queries": len(CASES),
        "repetitions_per_query": REPETITIONS,
        "baseline_tokens_total": baseline_total,
        "optimized_tokens_total": optimized_total,
        "token_reduction_percent": round(100 * (baseline_total - optimized_total) / baseline_total, 4),
        "baseline_accuracy_proxy": mean(r["baseline_accuracy_proxy"] for r in rows),
        "optimized_accuracy_proxy": mean(r["optimized_accuracy_proxy"] for r in rows),
        "baseline_completeness": mean(r["baseline_completeness"] for r in rows),
        "optimized_completeness": mean(r["optimized_completeness"] for r in rows),
        "selection_consistency": mean(r["selection_consistency"] for r in rows),
        "decision_consistency": mean(r["decision_consistency"] for r in rows),
        "mean_knapsack_time_ms": round(mean(times) * 1000, 6),
        "std_knapsack_time_ms": round(pstdev(times) * 1000, 6),
        "important_note": "Accuracy is an oracle coverage proxy, not measured LLM accuracy. Tokens are synthetic costs. Runtime measures only Knapsack, not model inference.",
    }
    return rows, summary


def main():
    rows, summary = run_experiment()
    out_dir = Path(__file__).resolve().parent
    with (out_dir / "m2_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "m2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("=" * 72)
    print("M2 - ROBUSTNESS AND DETERMINISTIC EVALUATION (SYNTHETIC)")
    print("=" * 72)
    for r in rows:
        print(f"{r['case_id']} | gold={r['reference_answer']:<7} | predicted={r['optimized_prediction']:<20} | "
              f"tokens={r['optimized_tokens']:>3}/{r['baseline_tokens']} | completeness={r['optimized_completeness']:.2f}")
    print("-" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nSaved: m2_results.csv and m2_summary.json")


if __name__ == "__main__":
    main()
