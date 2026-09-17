from pathlib import Path
import tempfile
import unittest

from probekv.durable_event_journal import journal_path, archive_journal
from probekv.v8_schema10_event_log import OnlineEventLog, read_events


class JournalTests(unittest.TestCase):
    def test_retains_durable_source_and_exact_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = journal_path(root, root / 'system-journal')
            binding = {k: 'test' for k in ('code_commit', 'patch_sha256', 'model_signature',
                'config_sha256', 'initial_state_sha256', 'runtime_measurement_sha256', 'dispatch')}
            log = OnlineEventLog(source, binding=binding)
            log.append('request_started', 'r', {'test': True})
            destination = root / 'events.jsonl'
            audit = archive_journal(source, destination, binding=binding)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            self.assertEqual(len(read_events(destination, binding=binding)), 1)
            self.assertTrue(audit['request_fsync_enabled'])
            with self.assertRaises(FileExistsError):
                archive_journal(source, destination, binding=binding)
            source.write_text('broken\n')
            with self.assertRaises(ValueError):
                archive_journal(source, root / 'bad.jsonl', binding=binding)

    def test_requires_fresh_absolute_directory(self):
        with self.assertRaises(ValueError):
            journal_path('/unused', 'relative')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                journal_path(tmp, tmp)
