import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probekv.v8_schema10_execution import digest_json
from scripts.server.verify_schema10_gpu_handoff import verify


class GpuHandoffContractTests(unittest.TestCase):
    def test_incomplete_local_blueprint_is_explicitly_not_ready(self):
        manifest = {"protocol_version": 8, "schema_version": 10,
                    "max_integrated_concurrency": 1, "binding": {},
                    "paper_evidence": False, "locked_test_accessed": False}
        manifest["manifest_sha256"] = digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = verify(path, repository=Path.cwd())
        self.assertFalse(result["ready_for_gpu_sentinel"])
        self.assertIn("native_runtime_attachment_missing", result["errors"])

    def test_hash_mismatch_and_old_checkout_fail_closed(self):
        manifest = {"protocol_version": 8, "schema_version": 10,
                    "max_integrated_concurrency": 1, "binding": {"code_commit": "a"*40},
                    "paper_evidence": False, "locked_test_accessed": False,
                    "native_runtime": {}}
        manifest["manifest_sha256"] = digest_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("subprocess.check_output", return_value="b"*40 + "\n"):
                result = verify(path, repository=Path.cwd())
        self.assertIn("checkout_sha_mismatch", result["errors"])
