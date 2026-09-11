"""A code SHA cannot attest to untracked executable inputs."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("handoff_cli", Path(__file__).resolve().parents[1] / "scripts/prepare_schema10_single_request_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffCleanlinessTests(unittest.TestCase):
    def test_untracked_source_blocks_handoff(self):
        with patch.object(handoff, "git", side_effect=["", "src/probekv/new.py"]):
            self.assertTrue(handoff.checkout_changes())

    def test_tracked_changes_block_handoff(self):
        with patch.object(handoff, "git", side_effect=[" M src/probekv/old.py", ""]):
            self.assertTrue(handoff.checkout_changes())

    def test_unrelated_bundle_does_not_count_as_source(self):
        with patch.object(handoff, "git", return_value="") as git:
            self.assertFalse(handoff.checkout_changes())
        self.assertEqual(git.call_args.args[4:], ("src", "scripts", "tests", "configs", "patches", "docs"))

    def test_output_path_stays_inside_artifacts_on_python38(self):
        root = Path("artifacts").resolve()
        self.assertTrue(handoff._under(root / "new", root))
        self.assertFalse(handoff._under(root.parent / "other", root))


if __name__ == "__main__":
    unittest.main()
