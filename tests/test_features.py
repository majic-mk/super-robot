import unittest

from probekv.contracts import ProbeObservation
from probekv.features import (
    CacheCraftMetadata,
    cache_craft_style_score,
    combined_feature_vector,
    raw_drift_score,
)


class FeatureTests(unittest.TestCase):
    def test_metadata_baseline_and_combined_vector(self):
        observation = ProbeObservation(
            "c", "s", 2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1
        )
        metadata = CacheCraftMetadata(0.5, 0.6, 0.7, 0.8)
        self.assertTrue(0 <= cache_craft_style_score(metadata) <= 1)
        self.assertEqual(len(combined_feature_vector(observation, metadata)), 8)
        self.assertGreater(raw_drift_score(observation), 0)


if __name__ == "__main__":
    unittest.main()
