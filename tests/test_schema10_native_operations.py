import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from probekv.v8_schema10_native_operations import (NativeOperationDispatcher, OperationSpec,
    NativeMeasurementSession, RegisteredOperation)


class NativeOperationContractTests(unittest.TestCase):
    def test_dispatcher_never_accepts_an_unbound_callback_or_spec(self):
        collector = NS(provenance={"actual": True})
        dispatcher = NativeOperationDispatcher(adapter=object(), collector=collector)
        with self.assertRaises(TypeError):
            dispatcher.run(None, reset=lambda: None, operation=lambda: {})
        with self.assertRaises(TypeError):
            dispatcher.run(OperationSpec("joint_future", {}), reset=None, operation=lambda: {})

    def test_operation_cells_are_unique_and_reproducible(self):
        specs = [OperationSpec("joint_future", {"segments": 1}),
                 OperationSpec("joint_future", {"segments": 2})]
        self.assertEqual(NativeOperationDispatcher.expected_cells(specs),
                         NativeOperationDispatcher.expected_cells(specs))
        with self.assertRaises(ValueError):
            NativeOperationDispatcher.expected_cells([specs[0], specs[0]])

    def test_cost_collector_is_not_replaced_by_a_fake_dispatcher(self):
        from probekv.v8_schema10_cost_collection import CudaCostCollector
        provenance = {k: "test" for k in ("model", "code", "patch", "gpu", "config", "timing_scope")}
        provenance.update(runtime_profile="test-profile")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
            CudaCostCollector(provenance=provenance)
        with patch("torch.cuda.is_available", side_effect=AssertionError("validate provenance before CUDA")), self.assertRaises(ValueError):
            CudaCostCollector(provenance={})

    def test_registered_session_rejects_missing_or_duplicate_cells(self):
        class Collector:
            provenance = {"actual": True}
            def measure(self, **kwargs):
                return {"origin": "real_cuda_execution", "fake_timing": False,
                        "row_sha256": "row"}
        d = NativeOperationDispatcher(adapter=object(), collector=Collector())
        session = NativeMeasurementSession(d)
        spec = OperationSpec("comparison_batch", {"k": 1})
        session.register(RegisteredOperation(spec, lambda: None, lambda: {}))
        self.assertEqual(len(session.run()), 1)
        with self.assertRaises(ValueError):
            session.register(RegisteredOperation(spec, lambda: None, lambda: {}))
