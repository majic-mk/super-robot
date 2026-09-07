import json
from pathlib import Path
import tempfile
import unittest

from scripts.server.build_schema10_premeasurement_blueprint import main


class BlueprintBuilderTests(unittest.TestCase):
    def test_builder_rejects_placeholder_audits_without_writing(self):
        data = {"trace_set": {}, "dispatches": {}, "binding": {
            "code_commit": "a"*40, "patch_sha256": "b"*64,
            "config_sha256": "c"*64, "model_signature": "model",
            "model_revision": "revision", "tokenizer_hash": "pending-tokenizer-audit"}}
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "input.json"
            output = Path(root) / "blueprint.json"
            source.write_text(json.dumps(data), encoding="utf-8")
            import sys
            old = sys.argv
            sys.argv = ["build", "--input", str(source), "--output", str(output)]
            try:
                with self.assertRaises(ValueError):
                    main()
            finally:
                sys.argv = old
            self.assertFalse(output.exists())
