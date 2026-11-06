"""Unit tests for the writer-based dataset split."""

import unittest

import numpy as np
import pandas as pd

from src.split_dataset import split


def make_metadata(writers=100, images_per_writer=3):
    """Return a small metadata frame with several images per writer."""
    return pd.DataFrame(
        [
            {"writer_id": writer, "image": f"{writer}-{image}"}
            for writer in range(writers)
            for image in range(images_per_writer)
        ]
    )


class SplitTests(unittest.TestCase):
    def test_sets_are_disjoint_and_cover_every_writer(self):
        metadata = make_metadata()
        train, validation, test = split(metadata, seed=12)

        groups = [set(frame["writer_id"]) for frame in (train, validation, test)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(groups[0] | groups[1] | groups[2], set(metadata["writer_id"]))
        self.assertEqual(
            sum(len(frame) for frame in (train, validation, test)), len(metadata)
        )

    def test_writer_counts_follow_proportions(self):
        train, validation, test = split(make_metadata(100), seed=12)

        self.assertEqual(train["writer_id"].nunique(), 70)
        self.assertEqual(validation["writer_id"].nunique(), 15)
        self.assertEqual(test["writer_id"].nunique(), 15)

    def test_same_seed_reproduces_the_assignment(self):
        first = split(make_metadata(), seed=12)
        second = split(make_metadata(), seed=12)
        for left, right in zip(first, second):
            self.assertTrue(left.equals(right))

    def test_different_seed_changes_the_assignment(self):
        default = split(make_metadata(), seed=12)[1]
        other = split(make_metadata(), seed=23)[1]
        self.assertFalse(default.equals(other))

    def test_rejects_invalid_settings(self):
        metadata = make_metadata()
        with self.assertRaises(ValueError):
            split(metadata, training=0.8)
        with self.assertRaises(ValueError):
            split(metadata, training=-0.1)
        with self.assertRaises(ValueError):
            split(metadata.assign(writer_id=np.nan), seed=12)


if __name__ == "__main__":
    unittest.main()
