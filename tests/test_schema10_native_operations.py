import unittest
from types import SimpleNamespace as NS

from probekv.v8_schema10_native_operations import NativeOperationDispatcher, OperationSpec


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
        with self.assertRaises(RuntimeError):
            CudaCostCollector(provenance={})
