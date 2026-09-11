import tempfile
from pathlib import Path
import unittest

from scripts.build_local_asset_readiness import build_readiness


class DualAssetReadinessTests(unittest.TestCase):
    def test_both_models_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ("mistral", "qwen"):
                directory = root / model
                directory.mkdir()
                for filename in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                    (directory / filename).write_text("{}", encoding="utf-8")
            partition = root / "partition.json"
            partition.write_text('{"kind":"source_policy_capture_partition_v1","locked_test_accessed":false,"entries":[{"request_id":"q"}]}', encoding="utf-8")
            report = build_readiness({"mistral": root / "mistral", "qwen": root / "qwen"}, partition)
            self.assertTrue(report["all_models_ready"])

    def test_one_missing_model_blocks_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_readiness({"mistral": root / "m", "qwen": root / "q"}, root / "p")
            self.assertFalse(report["all_models_ready"])
