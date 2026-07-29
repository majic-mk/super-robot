import unittest

from probekv.cacheblend_closed_loop_runtime import (
    CacheBlendClosedLoopRuntime,
)
from probekv.closed_loop_audit import audit_cacheblend_closed_loop
from probekv.contracts import CostAccountingPolicy
from probekv.orchestration import TwoStageReuseController

from tests.test_cacheblend_closed_loop_runtime import (
    FakeClosedLoopEngine,
    selected,
)
from tests.helpers import canonical_source


def run_record(repair_ms):
    source = canonical_source()
    runtime = CacheBlendClosedLoopRuntime(
        FakeClosedLoopEngine(repair_ms=repair_ms),
        {source.source_id: source},
        total_layers=32,
    )
    return TwoStageReuseController(
        cost_accounting_policy=CostAccountingPolicy.UNIFIED_COMPONENTS_V1
    ).execute(selected(), runtime).to_audit_record()


class ClosedLoopAuditTests(unittest.TestCase):
    def test_accept_and_refined_reject_are_auditable(self):
        result = audit_cacheblend_closed_loop(
            [run_record(20), run_record(80)]
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["rejected"], 1)

    def test_source_switch_is_rejected(self):
        record = run_record(20)
        record["runtime_selected_source_id"] = "s2"
        result = audit_cacheblend_closed_loop([record])
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("another Source" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
