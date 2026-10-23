# LLM Return Policy Token Optimization

**Midterm 1 — M1/M2 implemented; M3/M4 planned, with exploratory prototypes.**

Algorithmic selection of return-policy fragments under a context budget using
0/1 Knapsack dynamic programming. All current policy examples, relevance scores,
and fragment costs are **synthetic**. **No LLM API is called.**

## Requirements

Python 3.10+; no external Python packages required.

## Quick start

```bash
python3 m1_knapsack_experiment.py
python3 m2_robustness_experiment.py
python3 -m unittest discover -s tests -v
python3 benchmark.py --repetitions 30
python3 scalability.py --seed 42 --trials 5
```

`m2_robustness_experiment.py` writes `m2_results.csv` and `m2_summary.json`
in the repository root. Copies of the original reported outputs are in `results/`.
The exploratory benchmark writes `results/benchmark.csv`; scaling writes
`results/scalability.csv`. Timings depend on the local machine.

## Repository map

| Path | Purpose | Status |
| --- | --- | --- |
| `m1_knapsack_experiment.py` | Original M1 dynamic-programming experiment | Implemented |
| `m2_robustness_experiment.py` | Original M2, 10 synthetic queries x 30 repetitions | Implemented |
| `src/optimizer.py` | Reusable DP, greedy ratio, top-relevance, baseline | Exploratory extension |
| `src/adaptive.py` | Exact minimum-cost threshold with prerequisites (small n) | M3 prototype, not validated |
| `benchmark.py` | Compare four selectors on same synthetic fixture | M4 prototype, not LLM benchmark |
| `scalability.py` | Scale n and B with fixed RNG seeds | Exploratory extension |
| `tests/` | Correctness and regression tests | Implemented |
| `paper/` | IEEE LaTeX and figures | Midterm document |
| `results/` | Saved original M2 outputs | Preliminary evidence |

## Existing M1/M2 evidence

- M1: 8 rules, budget 350, synthetic baseline 865 cost units,
  selected 310 units (64.16% reduction).
- M2: 10 fixed synthetic queries, 30 repeated selections per query,
  baseline 8,650 vs selected 3,090 cost units (64.28% reduction).
- Required-rule coverage and deterministic oracle proxy: 100% on this
  constructed fixture. The scoring uses the required-rule oracle, so this
  is **not** an independent retrieval or LLM accuracy result.
- Costs are illustrative units, **not tokenizer-measured tokens**.
- Runtime is selector-only, **not end-to-end inference latency**.

## Algorithmic guarantees and limitations

0/1 DP optimizes additive relevance exactly under integer budget B,
using O(nB) time and O(nB) space with reconstruction. This does not
prove correctness of LLM decisions. `src/adaptive.py` enforces rule
prerequisites and a numeric relevance threshold by exhaustive search
(O(2^n) subsets), intended for small test instances only. `baseline`
intentionally ignores the budget to represent sending the full manual.

## Reproducibility and next steps

1. **M1**: maintain the DP implementation and unit tests.
2. **M2**: preserve synthetic case IDs, gold labels, required rules,
   fixed inputs and 30 repetitions; document all output files.
3. **M3 (planned)**: replace oracle-informed relevance with independent
   BM25/embedding retrieval, add a real tokenizer, dependency handling,
   adaptive budgets and abstention/escalation; measure quality with
   independent human-verified answers.
4. **M4 (planned)**: version the real policy, use >=200 stratified
   independently annotated cases, run full-manual/top-k/greedy/DP/adaptive
   with the same model, prompts and sampling controls; log input/output
   tokens, monetary cost, latency, memory, accuracy, completeness,
   consistency and scalability; report paired confidence intervals.

