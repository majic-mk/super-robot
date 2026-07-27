import json
import tempfile
import unittest
from pathlib import Path

from probekv.config import ExperimentConfig
from probekv.local_e1e2 import run_local_e1e2
from probekv.manifest import (
    case_from_mapping,
    manifest_digest,
    manifest_case_from_row,
    synthetic_manifest,
    token_content_hash,
    validate_manifest,
)


class ManifestTests(unittest.TestCase):
    def test_token_hash_is_order_sensitive(self):
        self.assertNotEqual(token_content_hash([1, 2]), token_content_hash([2, 1]))

    def test_synthetic_manifest_is_balanced_and_isolated(self):
        cases = synthetic_manifest(20, 20260726)
        validate_manifest(cases)
        self.assertEqual({case.split for case in cases}, {"train", "calibration", "test"})
        self.assertEqual(len({case.case_id for case in cases}), 20)
        self.assertEqual(len(manifest_digest(cases)), 64)
        self.assertEqual(manifest_case_from_row(cases[0].to_row()), cases[0])

    def test_mapping_requires_model_token_ids(self):
        case = case_from_mapping(
            {
                "case_id": "c1",
                "dataset": "fixture",
                "document_id": "d1",
                "segment_text": "same segment",
                "segment_token_ids": [10, 11, 12],
                "current_context": "current",
                "historical_contexts": ["a", "b", "c", "d"],
            },
            "model@revision",
        )
        self.assertEqual(len(case.sources), 4)
        self.assertEqual(case.model_signature, "model@revision")

    def test_current_context_cannot_be_a_historical_source(self):
        with self.assertRaisesRegex(ValueError, "current context"):
            case_from_mapping(
                {
                    "case_id": "c1",
                    "dataset": "fixture",
                    "document_id": "d1",
                    "segment_text": "same segment",
                    "segment_token_ids": [10, 11, 12],
                    "current_context": "current",
                    "historical_contexts": ["current", "b", "c", "d"],
                },
                "model@revision",
            )

    def test_same_content_cannot_cross_splits_under_different_documents(self):
        first = synthetic_manifest(3, 20260726)[0]
        from dataclasses import replace

        second = replace(
            first,
            case_id="other",
            document_id="other-document",
            group_id="other-group",
            split="test" if first.split != "test" else "train",
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_manifest([first, second])


class LocalE1E2Tests(unittest.TestCase):
    def test_pipeline_writes_full_non_paper_audit_set_and_resumes(self):
        config = ExperimentConfig.from_mapping(
            {
                "name": "test-e1e2",
                "evidence_class": "local_simulation",
                "seed": 20260726,
                "cases": 20,
                "total_layers": 32,
                "online_kmax": 4,
                "gamma": 0.8,
                "probe_checkpoints": [1, 2, 3, 4, 5, 6, 7, 8],
                "repair_ratios": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
                "output_dir": "unused",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = run_local_e1e2(config, output)
            self.assertFalse(summary["paper_evidence"])
            self.assertTrue(summary["thresholds_frozen_before_test"])
            self.assertIn("selected_quality_budget_coverage", summary)
            for filename in (
                "manifest.json",
                "case_manifest.jsonl",
                "ratio_measurements.jsonl",
                "safe_budget_labels.jsonl",
                "probe_observations.jsonl",
                "calibration_report.json",
                "decisions.jsonl",
                "summary.json",
                "ledger.json",
            ):
                self.assertTrue((output / filename).exists(), filename)
            resumed = run_local_e1e2(config, output, resume=True)
            self.assertTrue(resumed["resumed"])
            ledger = json.loads((output / "ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["stages"]["evaluate"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()
