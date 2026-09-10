import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from probekv.source_policy_replay import (
    PROVENANCE_FIELDS, build_observation, observe_depth_k, replay_observation,
    validate_observation,
)
from probekv.v8_schema10_execution import digest_json


def observation(count=4, eligible_count=None):
    current = torch.ones(20, 2, 4, dtype=torch.bfloat16)
    d1 = {"s%d" % i: current + (i + 1) / 8 for i in range(count)}
    # D1's worst Source is D2's best: pruning must report the lost winner.
    d2 = {"s%d" % i: current + (count - i) / 8 for i in range(count)}
    provenance = {key: "test-" + key for key in PROVENANCE_FIELDS}
    for key in PROVENANCE_FIELDS:
        if key.endswith(("sha256", "digest")):
            provenance[key] = digest_json(key)
    provenance["code_commit"] = "a" * 40
    return build_observation(provenance=provenance, absolute_positions=range(128, 148),
        correctness_eligible_source_ids=["s%d" % i for i in range(eligible_count or count)],
        depth_observations=[observe_depth_k(current, d1, completed_depth=1),
                            observe_depth_k(current, d2, completed_depth=2)])


def reseal(value):
    value["observation_sha256"] = digest_json({k: v for k, v in value.items() if k != "observation_sha256"})


class SourcePolicyReplayTests(unittest.TestCase):
    def test_full_cohort_shadow_exposes_lost_winner(self):
        value = observation()
        before = copy.deepcopy(value)
        result = replay_observation(value)
        self.assertEqual(value, before)
        self.assertEqual(len(result["cells"]), 4)
        for cell in result["cells"]:
            audit = cell["depth2_pruning_audit"]
            if cell["depth2_keep_fraction"] == .5:
                self.assertFalse(audit["oracle_winner_retained"])
                self.assertGreater(audit["absolute_residual_regret"], 0)
                self.assertEqual(cell["ranked_source_id"], "s1")
            else:
                self.assertTrue(audit["oracle_winner_retained"])
                self.assertEqual(audit["absolute_residual_regret"], 0)
                self.assertEqual(cell["ranked_source_id"], "s3")
            self.assertIsNone(cell["qa_passed"])
            self.assertIsNone(cell["pruned_execution_comparison_ms"])
            self.assertFalse(cell["source_lock_performed"])
        self.assertEqual(result, replay_observation(value))

    def test_true_single_and_budget_truncated_single_remain_distinct(self):
        for cell in replay_observation(observation(1))["cells"]:
            self.assertEqual(cell["ranking_status"], "SINGLE_CORRECTNESS_ELIGIBLE_SOURCE")
            self.assertEqual(cell["ranked_source_id"], "s0")
            self.assertIsNone(cell["margin"])
        for cell in replay_observation(observation(1, 16))["cells"]:
            self.assertEqual(cell["ranking_status"], "INSUFFICIENT_RANKING_COVERAGE")
            self.assertIsNone(cell["ranked_source_id"])
            self.assertIsNone(cell["margin"])
            self.assertFalse(cell["ranking_scope_complete"])

    def test_partial_cohort_is_never_complete_inventory(self):
        for cell in replay_observation(observation(4, 16))["cells"]:
            self.assertFalse(cell["ranking_scope_complete"])
            self.assertEqual(cell["shortlist"]["correctness_eligible_k"], 16)

    def test_missing_d2_sources_cannot_be_a_shadow(self):
        value = observation()
        del value["depth_observations"][1]["sources"]["s3"]
        reseal(value)
        with self.assertRaisesRegex(ValueError, "entire d1 cohort"):
            replay_observation(value)

    def test_tampered_observation_rejected(self):
        value = observation()
        value["depth_observations"][0]["sources"]["s0"]["normalized_k_drifts"][0] = 2.
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            replay_observation(value)

    def test_depth_geometry_and_position_contracts(self):
        original = observation()
        changes = [lambda v: v["depth_observations"][0].update(completed_depth=0),
                   lambda v: v["depth_observations"][1].update(k_observation_layer_1based=2),
                   lambda v: v["depth_observations"][1].update(geometry=[20, 1, 4]),
                   lambda v: v["absolute_positions"].append(148),
                   lambda v: v["absolute_positions"].__setitem__(0, True),
                   lambda v: v["provenance"].update(code_commit="short")]
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            with self.assertRaises(ValueError):
                reseal(value)
                validate_observation(value)

    def test_nonfinite_and_short_drift_rows_rejected(self):
        for bad in (float("nan"), float("inf"), -1, True):
            value = observation()
            value["depth_observations"][1]["sources"]["s0"]["normalized_k_drifts"][0] = bad
            with self.assertRaises(ValueError):
                reseal(value)
                validate_observation(value)

    def test_sixteenth_source_can_win_and_pruned_loss_is_not_hidden(self):
        cells = replay_observation(observation(16))["cells"]
        for cell in cells:
            if cell["depth2_keep_fraction"] == 1:
                self.assertEqual(cell["ranked_source_id"], "s15")
            else:
                self.assertEqual(len(cell["shortlist"]["retained_source_ids"]), 8)
                self.assertFalse(cell["depth2_pruning_audit"]["oracle_winner_retained"])

    def test_flat_d1_keeps_all_tied_sources(self):
        value = observation(16)
        for source in value["depth_observations"][0]["sources"].values():
            source["normalized_k_drifts"] = [.1] * 20
        reseal(value)
        for cell in replay_observation(value)["cells"]:
            self.assertEqual(len(cell["shortlist"]["retained_source_ids"]), 16)
            self.assertTrue(cell["depth2_pruning_audit"]["oracle_winner_retained"])

    def test_short_rows_and_unknown_sources_rejected(self):
        value = observation()
        value["depth_observations"][1]["sources"]["s0"]["normalized_k_drifts"].pop()
        reseal(value)
        with self.assertRaises(ValueError):
            validate_observation(value)
        value = observation()
        value["depth_observations"][1]["sources"]["foreign-source"] = value["depth_observations"][1]["sources"].pop("s0")
        reseal(value)
        with self.assertRaises(ValueError):
            validate_observation(value)

    def test_tensor_observation_matches_scalar_norm_without_mutation(self):
        current = torch.ones(20, 2, 4, dtype=torch.bfloat16)
        source = current * 1.5
        before = source.clone()
        result = observe_depth_k(current, {"one": source}, completed_depth=1)
        self.assertTrue(torch.equal(before, source))
        self.assertEqual(result["k_observation_layer_1based"], 2)
        for drift in result["sources"]["one"]["normalized_k_drifts"]:
            self.assertAlmostEqual(drift, .5, places=6)
        with self.assertRaises(ValueError):
            observe_depth_k(current.float(), {"one": source}, completed_depth=1)
        with self.assertRaises(ValueError):
            observe_depth_k(current, {"one": source}, completed_depth=0)

    def test_cpu_or_declared_hook_observation_never_qualifies_gpu(self):
        value = observation()
        value["evidence_origin"] = "native_hook_diagnostic"
        reseal(value)
        self.assertFalse(replay_observation(value)["gpu_runtime_qualified"])
        value["gpu_runtime_qualified"] = True
        reseal(value)
        with self.assertRaises(ValueError):
            replay_observation(value)

    def test_cli_checks_bytes_and_retains_previous_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "observations.json"
            target = Path(directory) / "report.json"
            source.write_text(json.dumps(observation()), encoding="utf-8")
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            command = [sys.executable, str(root / "scripts/replay_source_policy_observations.py"),
                       "--input", str(source), "--input-sha256", sha, "--output", str(target)]
            env = dict(os.environ, PYTHONPATH=str(root / "src"))
            first = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            contents = target.read_bytes()
            second = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(target.read_bytes(), contents)
            command[-1] = str(Path(directory) / "bad.json")
            command[-3] = "0" * 64
            bad = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse(Path(command[-1]).exists())


if __name__ == "__main__":
    unittest.main()
