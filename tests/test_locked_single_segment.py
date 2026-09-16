import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/server/run_locked_single_segment.py'
spec = importlib.util.spec_from_file_location('locked_single_segment', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LockedEnvironmentTests(unittest.TestCase):
    def test_matched_executor_includes_resident_source_prerequisites(self):
        args = runner.matched_executor_args(20)
        for flag in ('--matched-executor-only', '--matched-repair-backends',
                     '--cacheblend-loop-control', '--gpu-hot-cache', '--cost-probe'):
            self.assertIn(flag, args)
        self.assertEqual(args[-2:], ['--backend-repeats', '20'])
        for invalid in (0, 21):
            with self.assertRaises(ValueError):
                runner.matched_executor_args(invalid)

    def test_rejects_import_from_old_editable_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'selected'
            self.assertEqual(runner.require_origin(root / 'vllm/__init__.py', root),
                             str((root / 'vllm/__init__.py').resolve()))
            with self.assertRaisesRegex(RuntimeError, 'unexpected imported'):
                runner.require_origin(Path(d) / 'old/vllm/__init__.py', root)

    def test_rehashes_actual_assets_instead_of_trusting_complete_flag(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'config.json'
            f.write_text('{}')
            audit = dict(complete=True, files={'config.json': runner.file_sha(f)},
                         snapshot_path=d, tokenizer_assets_sha256='test-only')
            runner.verify_assets(audit)
            f.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                runner.verify_assets(audit)

    def test_incomplete_assets_and_relative_escape_fail_closed(self):
        with self.assertRaises(ValueError):
            runner.verify_assets({'complete': True})
        with self.assertRaisesRegex(ValueError, 'relative path'):
            runner.verify_assets(dict(complete=True, files={'../secret': 'bad'},
                                      snapshot_path='.', tokenizer_assets_sha256='test-only'))

    def test_cost_probe_is_not_a_default_cli_side_effect(self):
        # Structural check complements asset/path tests; no synthetic GPU pass.
        source = path.read_text(encoding='utf-8')
        self.assertIn("if args.execute:", source)
        self.assertIn("verify_cacheblend_patch.py", source)
        self.assertNotIn("pip install", source)
        self.assertNotIn("git reset", source)
        self.assertIn("launch.extend(['--kv-layout-mode', args.kv_layout_mode])", source)
        self.assertIn("choices=('legacy', 'packed_slice'), default='legacy'", source)


if __name__ == '__main__':
    unittest.main()
