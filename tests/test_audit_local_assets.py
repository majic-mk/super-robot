import json
import tempfile
from pathlib import Path
import unittest

from scripts.audit_local_assets import audit_asset_root


class LocalAssetAuditTests(unittest.TestCase):
    def test_missing_assets_have_null_hash_and_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit_asset_root(tmp)
            self.assertFalse(report["ready"])
            self.assertIsNone(report["tokenizer"]["tokenizer.json"]["sha256"])

    def test_real_assets_and_partition_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            partition = root / "partition.json"
            partition.write_text(json.dumps({"kind": "source_policy_capture_partition_v1",
                "locked_test_accessed": False, "entries": [{"request_id": "q"}]}), encoding="utf-8")
            report = audit_asset_root(root, partition=partition)
            self.assertTrue(report["ready"])
            self.assertEqual(len(report["development_partition"]["sha256"]), 64)

    def test_locked_partition_never_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            partition = root / "partition.json"
            partition.write_text(json.dumps({"kind": "source_policy_capture_partition_v1",
                "locked_test_accessed": True, "entries": [{"request_id": "q"}]}), encoding="utf-8")
            self.assertFalse(audit_asset_root(root, partition=partition)["ready"])
