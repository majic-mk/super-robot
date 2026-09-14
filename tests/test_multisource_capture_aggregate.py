import json
import tempfile
import unittest
from pathlib import Path

import torch

from probekv.source_policy_replay import (
    PROVENANCE_FIELDS, build_observation, observe_depth_k, replay_observation,
)
from probekv.v8_schema10_execution import digest_json
from scripts.server.aggregate_multisource_capture_matrix import aggregate


def capture(root, index, d1, d2):
    child = root / str(index)
    child.mkdir()
    provenance = {key: "test-" + key for key in PROVENANCE_FIELDS}
    for key in PROVENANCE_FIELDS:
        if key.endswith(("sha256", "digest")):
            provenance[key] = digest_json(key)
    provenance["code_commit"] = "a" * 40
    provenance["request_id"] = "request-" + str(index)
    current = torch.ones(20, 2, 4, dtype=torch.bfloat16)
    obs = build_observation(provenance=provenance, absolute_positions=range(20),
        correctness_eligible_source_ids=sorted(d1), depth_observations=[
            observe_depth_k(current, {s: current + v for s, v in scores.items()},
                            completed_depth=d)
            for d, scores in ((1, d1), (2, d2))])
    (child / "observation.json").write_text(json.dumps(obs), encoding="utf-8")
    (child / "replay.json").write_text(json.dumps(replay_observation(obs)), encoding="utf-8")
    return child


class MultiSourceCaptureAggregateTests(unittest.TestCase):
    def test_rank_changes_do_not_prove_qa_or_net_gain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = capture(root, 0, {"a": .1, "b": .2}, {"a": .2, "b": .1})
            second = capture(root, 1, {"a": .2, "b": .1}, {"a": .2, "b": .1})
            report = aggregate([first, second])
            self.assertEqual(report["capture_count"], 2)
            self.assertEqual(report["d1_d2_disagreement_count"], 1)
            self.assertEqual(report["candidate_pool_count"], 1)
            self.assertTrue(report["same_pool_cross_request_rank_change_observed"])
            self.assertIsNone(report["multisource_complementarity_observed"])
            self.assertFalse(report["reuse_gain_measured"])
            self.assertEqual(report["evidence_origin"], "cpu_interface_test")

    def test_different_pools_never_prove_complementarity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = capture(root, 0, {"a": .1, "b": .2}, {"a": .1, "b": .2})
            second = capture(root, 1, {"c": .1, "d": .2}, {"c": .1, "d": .2})
            report = aggregate([first, second])
            self.assertEqual(report["distinct_d1_winners"], 2)
            self.assertEqual(report["candidate_pool_count"], 2)
            self.assertFalse(report["same_pool_cross_request_rank_change_observed"])
            self.assertIsNone(report["multisource_complementarity_observed"])

    def test_missing_raw_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = capture(Path(td), 0, {"a": .1}, {"a": .1})
            (p / "observation.json").unlink()
            with self.assertRaises(FileNotFoundError):
                aggregate([p])

    def test_tampered_replay_rejected_even_if_resealed(self):
        with tempfile.TemporaryDirectory() as td:
            p = capture(Path(td), 0, {"a": .1}, {"a": .1})
            path = p / "replay.json"
            value = json.loads(path.read_text())
            value["cells"][0]["depth1_scores"]["a"] = 0
            value["report_sha256"] = digest_json({k: v for k, v in value.items() if k != "report_sha256"})
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw observation"):
                aggregate([p])

    def test_duplicate_capture_cannot_inflate_sample_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = capture(Path(td), 0, {"a": .1}, {"a": .1})
            with self.assertRaisesRegex(ValueError, "duplicate observation"):
                aggregate([p, p])


if __name__ == "__main__":
    unittest.main()
