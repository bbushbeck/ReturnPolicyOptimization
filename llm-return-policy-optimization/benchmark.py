"""M4 prototype: reproducible synthetic selector benchmark (NO LLM).

Usage: python benchmark.py --repetitions 30 --output results/benchmark.csv
"""
import argparse
import csv
from pathlib import Path
from statistics import mean
from time import perf_counter
from src.fixtures import CASES, BUDGET, rules_for_case
from src.optimizer import baseline, greedy_ratio, knapsack, top_relevance, cost

METHODS = {"full_manual": baseline, "top_relevance": top_relevance,
           "greedy_ratio": greedy_ratio, "knapsack": knapsack}


def run(repetitions: int = 30):
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rows = []
    for case in CASES:
        rules = rules_for_case(case)
        required = set(case["required"])
        for method, selector in METHODS.items():
            samples, elapsed = [], []
            for _ in range(repetitions):
                t0 = perf_counter()
                selected = selector(rules, BUDGET)
                elapsed.append((perf_counter() - t0) * 1000)
                names = tuple(r.name for r in selected)
                samples.append((names, cost(selected)))
            selected_names, selected_cost = samples[0]
            coverage = len(set(selected_names) & required) / len(required)
            rows.append({"case_id": case["id"], "method": method,
                         "budget": BUDGET, "synthetic_cost": selected_cost,
                         "relevance": sum(r.relevance for r in rules if r.name in selected_names),
                         "required_coverage": coverage,
                         "oracle_proxy_accuracy": int(required.issubset(selected_names)),
                         "selection_consistency": sum(s == samples[0] for s in samples) / repetitions,
                         "mean_selector_ms": round(mean(elapsed), 6),
                         "repetitions": repetitions})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--output', default='results/benchmark.csv')
    args = parser.parse_args()
    rows = run(args.repetitions)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print('Saved', path, '|', len(rows), 'case-method comparisons')
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        print(f"{method:15s} avg cost={mean(r['synthetic_cost'] for r in group):6.1f} "
              f"coverage={mean(r['required_coverage'] for r in group):.3f} "
              f"oracle proxy={mean(r['oracle_proxy_accuracy'] for r in group):.3f}")
    print('WARNING: Costs are illustrative; oracle proxy is not LLM accuracy.')

if __name__ == '__main__':
    main()
