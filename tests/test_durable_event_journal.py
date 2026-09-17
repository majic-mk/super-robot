from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probekv.durable_event_journal import journal_path, archive_journal
from probekv.v8_schema10_event_log import OnlineEventLog, read_events


class JournalTests(unittest.TestCase):
    def test_transaction_durability_is_explicit_and_incomplete_resume_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            binding = {k: 'test' for k in ('code_commit', 'patch_sha256', 'model_signature',
                'config_sha256', 'initial_state_sha256', 'runtime_measurement_sha256', 'dispatch')}
            with self.assertRaises(ValueError):
                OnlineEventLog(path, binding=binding, durability='request_finalized')
            binding['event_durability'] = 'request_finalized'
            with self.assertRaises(ValueError):
                OnlineEventLog(path, binding=binding)
            log = OnlineEventLog(path, binding=binding, durability='request_finalized')
            with patch('probekv.v8_schema10_event_log.os.fsync') as sync:
                log.append('request_started', 'r', {})
                log.append('request_completed', 'r', {})
                sync.assert_not_called()
                with self.assertRaises(ValueError):
                    OnlineEventLog(path, binding=binding, resume=True, durability='request_finalized')
                log.append('request_finalized', 'r', {})
                sync.assert_called_once()
            resumed = OnlineEventLog(path, binding=binding, resume=True, durability='request_finalized')
            with patch('probekv.v8_schema10_event_log.os.fsync') as sync:
                resumed.append('request_failed', 'bad', {})
                sync.assert_called_once()
            with self.assertRaises(ValueError):
                OnlineEventLog(path, binding=binding, resume=True, durability='request_finalized')

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
