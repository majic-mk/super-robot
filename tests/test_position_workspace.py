import unittest
from types import SimpleNamespace

from probekv.position_workspace import request_position_tensors, POSITION_WORKSPACE_KEY
from probekv.position_workspace import active_positions_strictly_increasing
from unittest.mock import patch
import torch as real_torch


class FakeTorch:
    long = "long"

    def __init__(self):
        self.calls = []

    def arange(self, n, **kwargs):
        return self.as_tensor(tuple(range(n)), **kwargs)

    def as_tensor(self, rows, **kwargs):
        value = SimpleNamespace(rows=tuple(rows), device=kwargs["device"])
        self.calls.append(value)
        return value


class PositionWorkspaceTests(unittest.TestCase):
    def test_owned_host_validation_avoids_device_scalar_check_in_inference_mode(self):
        m = {"org_seq_len": 10, "probekv_host_position_validation": True}
        with real_torch.inference_mode():
            a, _, _ = request_position_tensors(m, (2, 4, 9), (2, 4, 9), "cpu", real_torch)
            with patch.object(real_torch, "all", side_effect=AssertionError("device fence")):
                self.assertTrue(active_positions_strictly_increasing(m, a, real_torch))
            self.assertEqual(m["probekv_position_validation_audit"], {"owned_host_checks": 1})
            a[1] = 2
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                active_positions_strictly_increasing(m, a, real_torch)
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                request_position_tensors(m, (2, 4, 9), (2, 4, 9), "cpu", real_torch)

    def test_unbound_and_disabled_positions_keep_device_validation(self):
        for m in ({}, {"org_seq_len": 10, "probekv_host_position_validation": True}):
            self.assertFalse(active_positions_strictly_increasing(m, real_torch.tensor([2, 2]), real_torch))
            self.assertEqual(m["probekv_position_validation_audit"]["device_checks"], 1)
        m = {"org_seq_len": 10, "probekv_host_position_validation": True}
        a, _, _ = request_position_tensors(m, (4, 2), (4, 2), "cpu", real_torch)
        self.assertFalse(active_positions_strictly_increasing(m, a, real_torch))

    def test_unchanged_layers_allocate_indices_once(self):
        torch, metadata = FakeTorch(), {"org_seq_len": 960}
        active = tuple(range(256, 960))
        a, t, full = request_position_tensors(metadata, active, active, "cuda:0", torch)
        for _ in range(31):
            aa, tt, ff = request_position_tensors(metadata, active, active, "cuda:0", torch)
            self.assertIs(aa, a)
            self.assertIs(tt, t)
            self.assertIs(ff, full)
        self.assertIs(a, t)
        self.assertEqual(len(torch.calls), 2)
        self.assertEqual(a.rows, active)

    def test_shrink_reuses_previous_target_and_discards_old_rows(self):
        torch, metadata = FakeTorch(), {"org_seq_len": 10}
        before, after = tuple(range(2, 10)), (2, 5, 9)
        a, _, full = request_position_tensors(metadata, before, before, "cuda:0", torch)
        b, target, ff = request_position_tensors(metadata, before, after, "cuda:0", torch)
        self.assertIs(a, b)
        self.assertIs(full, ff)
        c, d, _ = request_position_tensors(metadata, after, after, "cuda:0", torch)
        self.assertIs(c, target)
        self.assertIs(c, d)
        self.assertEqual(len(metadata[POSITION_WORKSPACE_KEY]["rows"]), 1)
        self.assertEqual(len(torch.calls), 3)

    def test_device_prompt_and_request_reset_invalidate(self):
        torch, metadata = FakeTorch(), {"org_seq_len": 10}
        a, _, _ = request_position_tensors(metadata, (2, 3), (2, 3), "cuda:0", torch)
        b, _, _ = request_position_tensors(metadata, (2, 3), (2, 3), "cuda:1", torch)
        self.assertIsNot(a, b)
        metadata["org_seq_len"] = 12
        c, _, full = request_position_tensors(metadata, (2, 3), (2, 3), "cuda:1", torch)
        self.assertIsNot(b, c)
        self.assertEqual(full.rows, tuple(range(12)))
        metadata.pop(POSITION_WORKSPACE_KEY)
        d, _, _ = request_position_tensors(metadata, (2, 3), (2, 3), "cuda:1", torch)
        self.assertIsNot(c, d)
