import unittest
from itertools import combinations
from src.optimizer import Rule, knapsack, greedy_ratio, top_relevance, cost, value
from src.adaptive import minimum_cost_quality
from src.fixtures import CASES, BUDGET, rules_for_case
from benchmark import run

class TestSelectors(unittest.TestCase):
    def test_knapsack_optimality_bruteforce(self):
        rules = [Rule('a', 2, 3), Rule('b', 3, 4), Rule('c', 4, 5), Rule('d', 5, 8)]
        for budget in range(15):
            selected = knapsack(rules, budget)
            brute = max((value(group) for n in range(len(rules)+1)
                         for group in combinations(rules, n) if cost(group) <= budget), default=0)
            self.assertEqual(value(selected), brute)
            self.assertLessEqual(cost(selected), budget)

    def test_fixed_fixture(self):
        for case in CASES:
            selected = knapsack(rules_for_case(case), BUDGET)
            self.assertLessEqual(cost(selected), BUDGET)
            self.assertTrue(set(case['required']).issubset({r.name for r in selected}))

    def test_greedy_can_be_suboptimal(self):
        rules = [Rule('a', 10, 60), Rule('b', 20, 100), Rule('c', 30, 120)]
        self.assertLess(value(greedy_ratio(rules, 50)), value(knapsack(rules, 50)))

    def test_adaptive_dependencies(self):
        rules = [Rule('exception', 2, 9), Rule('condition', 3, 1), Rule('other', 5, 2)]
        selected = minimum_cost_quality(rules, 9, {'exception': {'condition'}})
        self.assertEqual({r.name for r in selected}, {'exception', 'condition'})
        self.assertIsNone(minimum_cost_quality(rules, 20))

    def test_benchmark_rows(self):
        rows = run(3)
        self.assertEqual(len(rows), len(CASES)*4)
        self.assertTrue(all(r['selection_consistency'] == 1 for r in rows))

if __name__ == '__main__': unittest.main()
