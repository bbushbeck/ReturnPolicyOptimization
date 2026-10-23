"""Import the original M2 synthetic fixture without changing its semantics."""
from m2_robustness_experiment import RULES, CASES, BUDGET, case_rules
from .optimizer import Rule


def rules_for_case(case):
    return [Rule(name=r["name"], cost=r["tokens"], relevance=r["relevance"])
            for r in case_rules(case)]
