import json
import tempfile
import unittest
from pathlib import Path

from probekv.config import load_config
from probekv.manifest import ManifestCase, ManifestSource, manifest_case_from_row, token_content_hash
from probekv.v8_simulation import run_v8_local_simulation


ROOT = Path(__file__).resolve().parents[1]


class V8ConfigTests(unittest.TestCase):
    def test_v8_configs_and_legacy_v7_load(self):
        causal = load_config(str(ROOT / "configs/local_system_v8_causal_wait.json"))
        immediate = load_config(str(ROOT / "configs/local_system_v8_immediate_staggered.json"))
        legacy = load_config(str(ROOT / "configs/local_system_v7_causal_wait.json"))
        self.assertEqual((causal.protocol_version, immediate.protocol_version), (8, 8))
        self.assertEqual(legacy.protocol_version, 7)
        self.assertFalse(causal.learned_selector_enabled)
        self.assertEqual(causal.fixed_repair_ratio, 0.15)
        mistral_h1 = load_config(str(ROOT / "configs/a800_h1_pilot_v8_mistral.json"))
        qwen_h1 = load_config(str(ROOT / "configs/a800_h1_pilot_v8_qwen.json"))
        self.assertEqual(mistral_h1.v8_execution_phase, "h1_offline_diagnostic")
        self.assertEqual(len(qwen_h1.repair_ratios), 9)
        self.assertIn(0.15, mistral_h1.repair_ratios)
        self.assertNotIn(0.16, mistral_h1.repair_ratios)
        mistral_raw = json.loads(
            (ROOT / "configs/a800_h1_pilot_v8_mistral.json").read_text(encoding="utf-8")
        )
        qwen_raw = json.loads(
            (ROOT / "configs/a800_h1_pilot_v8_qwen.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            (mistral_raw["h1_primary_completed_depth"], mistral_raw["first_reused_layer_1based"]),
            (4, 5),
        )
        self.assertEqual(
            (qwen_raw["h1_primary_completed_depth"], qwen_raw["first_reused_layer_1based"]),
            (3, 4),
        )

    def test_v8_rejects_learned_selector_and_conformal_policy(self):
        raw = json.loads(
            (ROOT / "configs/local_system_v8_causal_wait.json").read_text(encoding="utf-8")
        )
        for key, value in (
            ("learned_selector_enabled", True),
            ("quality_predictor_enabled", True),
            ("joint_quality_policy", "simultaneous_conformal"),
            ("online_kmax", 4),
        ):
            changed = dict(raw)
            changed[key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(str(path))

    def test_local_matrix_covers_arbitrary_segment_points(self):
        config = load_config(str(ROOT / "configs/local_system_v8_causal_wait.json"))
        result = run_v8_local_simulation(config)
        self.assertTrue(all(item["passed"] for item in result["gates"]))
        target = [
            item for item in result["rows"]
            if item["detected_segment_count"] == 37
            and item["stored_variants_per_segment"] == 16
        ]
        self.assertTrue(target)
        self.assertTrue(all(item["planned_segment_count"] == 37 for item in target))

    def test_v8_manifest_retains_canonical_parent_and_sixteen_variants(self):
        left, segment, right = (1, 2), (3, 4), (5, 6)
        case = ManifestCase(
            case_id="v8", dataset="d", document_id="doc", group_id="g",
            split="pilot", regime="controlled", model_signature="model",
            segment_text="segment", segment_token_ids=segment,
            content_hash=token_content_hash(segment), current_context="current",
            sources=tuple(
                ManifestSource("s%d" % index, "history-%d" % index, "ctx-%d" % index)
                for index in range(16)
            ),
            protocol_version=8, canonicalizer_signature="chunker",
            segment_provenance_id="provenance", reuse_content_key="reuse",
            canonical_parent_content_hash=token_content_hash(left + segment + right),
            canonical_parent_left_token_ids=left,
            canonical_parent_right_token_ids=right,
        )
        case.validate()
        self.assertEqual(manifest_case_from_row(case.to_row()), case)


if __name__ == "__main__":
    unittest.main()
