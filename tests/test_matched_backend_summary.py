import json
from pathlib import Path
import tempfile
import unittest

from probekv.matched_backend_summary import summarize_matched_backend
from probekv.v8_schema10_execution import digest_json


class MatchedBackendSummaryTests(unittest.TestCase):
    def fixture(self, root):
        def put(name, row, digest=False):
            if digest:
                row["raw_observation_sha256"] = digest_json(row)
            (root/name).write_text(json.dumps(row))
        put("r1-comparison.json", {"cb": {"passed": True}, "pb": {"passed": True}})
        for name in ("fixed15-equivalence.json", "boundary-equivalence-1.0.json", "boundary-equivalence-0.15.json"):
            put(name, {"passed": True})
        put("source-integrity.json", {"unchanged": True, "before": "digest", "after": "digest"})
        for prefix, arms in (("", ("dense", "cacheblend_loop", "probekv")), ("boundary-", ("cacheblend", "probekv"))):
            for i in range(4):
                for arm in arms:
                    put(f"{prefix}{i:02d}-{arm}.json", dict(arm=arm, repeat=i, warmup=i<2,
                        first_token_host_ms=1000 if i<2 else (50 if arm=="dense" else 30),
                        executor_host_ms=1000 if i<2 else 20, origin="real_cuda_execution", fake_timing=False,
                        cached_prefix_tokens=0, request_tokens_sha256="request", sampling_signature={"n":32},
                        source_id=None if arm=="dense" else "source", diagnostic_repair_ratio=.15,
                        external_repair_mask_sha256=None if arm=="dense" else "mask", boundary=2), True)
        return put

    def test_excludes_warmup_and_instrumented_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); put=self.fixture(root)
            put("instrumented-probekv.json", {"first_token_host_ms":99999})
            result=summarize_matched_backend(root, repeats=2)
            self.assertEqual(result["comparisons"]["setup_inclusive"]["dense"]["mean_ms"],50)
            self.assertEqual(result["comparisons"]["boundary_executor"]["probekv"]["n"],2)

    def test_failed_equivalence_missing_sample_bad_digest_and_mismatched_mask_rejected(self):
        for kind in ("equivalence", "missing", "digest", "mask", "boundary"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root=Path(d); put=self.fixture(root)
                p=root/"02-probekv.json"
                if kind=="equivalence": put("fixed15-equivalence.json", {"passed":False})
                elif kind=="missing": p.unlink()
                else:
                    row=json.loads(p.read_text()); row.pop("raw_observation_sha256")
                    if kind == "boundary": row.pop("boundary")
                    else: row["external_repair_mask_sha256"]="wrong"
                    put(p.name,row,kind in ("mask", "boundary"))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    summarize_matched_backend(root,repeats=2)
