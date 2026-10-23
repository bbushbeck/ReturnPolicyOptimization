"""Synthetic scaling of DP by number of fragments and budget, no LLM."""
import argparse
import csv
import random
from pathlib import Path
from statistics import mean
from time import perf_counter
from src.optimizer import Rule, knapsack


def run(seed=42, trials=5):
    rows = []
    for n in (8, 16, 32, 64):
        for budget in (150, 350, 700, 1400):
            times = []
            for t in range(trials):
                rng = random.Random(seed + n * 10000 + budget * 100 + t)
                rules = [Rule(f'R{i}', rng.randint(20, 150), rng.randint(1, 10)) for i in range(n)]
                start = perf_counter(); knapsack(rules, budget)
                times.append((perf_counter() - start) * 1000)
            rows.append({'n': n, 'budget': budget, 'trials': trials,
                         'mean_time_ms': round(mean(times), 6), 'seed': seed})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', default='results/scalability.csv')
    args = parser.parse_args()
    if args.trials <= 0: parser.error('trials must be positive')
    rows = run(args.seed, args.trials)
    dest = Path(args.output); dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print('Saved', dest, '|', len(rows), 'configurations')

if __name__ == '__main__': main()
