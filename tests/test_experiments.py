"""Unit tests for training settings and validation fingerprints."""

import unittest
from pathlib import Path

import pandas as pd

from src.experiments import DEFAULT_TRAINING, normalize_training, validation_fingerprint


def make_frame(root, labels=(0, 1)):
    """Build a minimal validation frame whose parent names identify the source."""
    return pd.DataFrame(
        {
            "image_path": [
                Path(root) / "ETL1C_07_unpack" / "00000.png",
                Path(root) / "ETL1C_08_unpack" / "00001.png",
            ],
            "writer_id": [101, 202],
            "label": list(labels),
        }
    )


class NormalizeTrainingTests(unittest.TestCase):
    def test_defaults_are_returned_for_empty_settings(self):
        self.assertEqual(normalize_training(), DEFAULT_TRAINING)
        self.assertEqual(normalize_training(None), DEFAULT_TRAINING)

    def test_legacy_seed_controls_both_stages(self):
        settings = normalize_training({"seed": 23})
        self.assertEqual(settings["split_seed"], 23)
        self.assertEqual(settings["training_seed"], 23)

    def test_explicit_seeds_override_defaults_independently(self):
        settings = normalize_training({"split_seed": 7, "training_seed": 34})
        self.assertEqual((settings["split_seed"], settings["training_seed"]), (7, 34))

    def test_rejects_mixed_seed_styles(self):
        with self.assertRaises(ValueError):
            normalize_training({"seed": 12, "split_seed": 12})

    def test_rejects_unknown_settings(self):
        with self.assertRaises(ValueError):
            normalize_training({"epochs": 5})


class ValidationFingerprintTests(unittest.TestCase):
    def test_ignores_the_dataset_root(self):
        self.assertEqual(
            validation_fingerprint(make_frame("/one/root")),
            validation_fingerprint(make_frame("/another/root")),
        )

    def test_detects_changed_labels(self):
        self.assertNotEqual(
            validation_fingerprint(make_frame("/root")),
            validation_fingerprint(make_frame("/root", labels=(1, 0))),
        )

    def test_is_order_dependent(self):
        frame = make_frame("/root")
        self.assertNotEqual(
            validation_fingerprint(frame),
            validation_fingerprint(frame.iloc[::-1].reset_index(drop=True)),
        )


if __name__ == "__main__":
    unittest.main()
