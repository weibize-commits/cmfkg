from __future__ import annotations

import unittest

from fera_kg.benchmark import load_labeled_split
from fera_kg.evaluation import FOUR_CLASS_LABELS, full_relation_verification_metrics, to_three_decision


class EvaluationTests(unittest.TestCase):
    def test_perfect_four_class_predictions(self) -> None:
        metrics = full_relation_verification_metrics(FOUR_CLASS_LABELS, FOUR_CLASS_LABELS)
        self.assertEqual(metrics["four_class"]["accuracy"], 1.0)
        self.assertEqual(metrics["four_class"]["macro_f1"], 1.0)
        self.assertEqual(metrics["three_decision"]["macro_f1"], 1.0)

    def test_four_to_three_decision_mapping(self) -> None:
        self.assertEqual(to_three_decision("SUPPORTED"), "PRESENT")
        self.assertEqual(to_three_decision("NOT_SUPPORTED"), "ABSENT")
        self.assertEqual(to_three_decision("INSUFFICIENT_CONTEXT"), "ABSTAIN")
        self.assertEqual(to_three_decision("ENTITY_OR_TYPE_ERROR"), "ABSTAIN")

    def test_regular_loader_rejects_test_split(self) -> None:
        with self.assertRaises(ValueError):
            load_labeled_split("retrospective_standard_test")


if __name__ == "__main__":
    unittest.main()
