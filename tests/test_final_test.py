"""Unit tests for the single-use test evaluation helpers."""

import unittest

import numpy as np

from src.final_test import _confusion_table, _per_class_table, wilson_interval

CHARACTERS = list("アイウエオカキクケコ")


class WilsonIntervalTests(unittest.TestCase):
    def test_interval_contains_the_estimate_and_stays_in_range(self):
        low, high = wilson_interval(9429, 10000)
        self.assertLessEqual(low, 0.9429)
        self.assertGreaterEqual(high, 0.9429)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_perfect_score_reaches_one(self):
        _, high = wilson_interval(10, 10)
        self.assertAlmostEqual(high, 1.0)

    def test_rejects_empty_sample(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


class PerClassTests(unittest.TestCase):
    def test_recall_and_error_counts_match_predictions(self):
        labels = np.array([0, 0, 1, 1, 2])
        predictions = np.array([0, 1, 1, 1, 1])
        table = _per_class_table(labels, predictions, CHARACTERS)
        by_character = table.set_index("Character")

        self.assertEqual(by_character.at["ア", "Images"], 2)
        self.assertEqual(by_character.at["ア", "Errors"], 1)
        self.assertAlmostEqual(by_character.at["ア", "Recall"], 0.5)
        self.assertEqual(by_character.at["イ", "Errors"], 0)
        self.assertEqual(int(table["Errors"].sum()), 2)

    def test_sorted_by_ascending_recall(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 0, 1, 0])
        table = _per_class_table(labels, predictions, CHARACTERS)
        self.assertEqual(table.iloc[0]["Character"], "イ")
        observed = table.loc[table["Images"] > 0]
        self.assertTrue(observed["Recall"].is_monotonic_increasing)


class ConfusionTests(unittest.TestCase):
    def test_excludes_correct_and_zero_cells(self):
        labels = np.array([0, 0, 2, 2, 2])
        predictions = np.array([0, 1, 1, 1, 2])
        table = _confusion_table(labels, predictions, CHARACTERS)

        self.assertEqual(
            list(table.columns), ["Actual", "Predicted", "Count"]
        )
        self.assertNotIn(("ア", "ア"), set(zip(table["Actual"], table["Predicted"])))
        self.assertTrue((table["Count"] > 0).all())
        self.assertEqual(int(table["Count"].sum()), 3)

    def test_sorted_by_descending_count(self):
        labels = np.array([0, 0, 0, 1])
        predictions = np.array([1, 1, 2, 1])
        table = _confusion_table(labels, predictions, CHARACTERS)
        self.assertEqual(table.iloc[0]["Count"], 2)
        self.assertTrue(table["Count"].is_monotonic_decreasing)


if __name__ == "__main__":
    unittest.main()
