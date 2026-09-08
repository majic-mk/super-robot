import json
import tempfile
import unittest
from pathlib import Path

import torch

from probekv.v8_schema10_execution import digest_json
from probekv.v8_schema10_storage import file_digest
from probekv.v8_schema10_native_evidence import read_signed_json, load_saved_logits


class NativeEvidenceFileTests(unittest.TestCase):
    def test_signed_json_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.json"
            row = {"token_ids": [1, 2]}
            row["digest"] = digest_json(row)
            path.write_text(json.dumps(row))
            self.assertEqual(read_signed_json(path, "digest"), row)
            row["token_ids"] = [3, 4]
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                read_signed_json(path, "digest")

    def test_logit_bytes_geometry_and_locator_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "logits.pt"
            torch.save(torch.ones(32, 8), path)
            row = dict(logits_path="logits.pt", logits_shape=[32, 8], logits_sha256=file_digest(path))
            self.assertEqual(load_saved_logits(root, row).shape, (32, 8))
            for changes in (dict(logits_path="../logits.pt"), dict(logits_shape=[33, 8]), dict(logits_sha256="bad")):
                with self.assertRaises(ValueError):
                    load_saved_logits(root, {**row, **changes})
            torch.save(torch.full((32, 8), float("nan")), path)
            row["logits_sha256"] = file_digest(path)
            with self.assertRaises(ValueError):
                load_saved_logits(root, row)
