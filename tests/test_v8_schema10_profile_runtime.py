import inspect
import unittest

from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v8_schema10_profile_runtime import Schema10DevelopmentCaseRuntime
from probekv.v8_schema10_profile_runtime import development_repair_measurement_plan
from probekv.v8_schema8_contracts import RepairRatioScope
from probekv.v8_schema8_repair import (
    SegmentLayerRepairRatio,
    validate_union_repair_ratio_plan,
)


class Schema10ProfileRuntimeContractTests(unittest.TestCase):
    def test_nonpaper_measurement_admission_is_explicit_executor_input(self):
        parameters = inspect.signature(
            RealCacheBlendA800Executor._reuse_generate
        ).parameters
        self.assertIn("force_nonpaper_measurement_admission", parameters)
        self.assertFalse(
            parameters["force_nonpaper_measurement_admission"].default
        )

    def test_development_repair_sweep_can_measure_legacy_checkpoint(self):
        source = inspect.getsource(Schema10DevelopmentCaseRuntime._generate)
        self.assertIn("force_nonpaper_measurement_admission=True", source)
        executor_source = inspect.getsource(
            RealCacheBlendA800Executor._reuse_generate
        )
        self.assertIn("first_probe not in {1, 2}", executor_source)
        self.assertIn("and not force_nonpaper_measurement_admission", executor_source)

    def test_development_ratio_has_explicit_nonfrozen_measurement_scope(self):
        for ratio in (0.10, 0.12, 0.20, 0.30, 0.50, 0.75):
            plan = development_repair_measurement_plan(
                segment_id="c0",
                first_reuse_layer=5,
                repair_ratio=ratio,
            )
            self.assertEqual(
                plan.scope, RepairRatioScope.DEVELOPMENT_PROFILE_MEASUREMENT
            )
            self.assertFalse(plan.profile_frozen)
            self.assertEqual(plan.ratios_for_layer(5), {"c0": ratio})

    def test_formal_fixed15_scope_still_rejects_arbitrary_ratio(self):
        with self.assertRaisesRegex(ValueError, "fixed15 requires 0.15"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.UNIFORM_FIXED,
                rows=(SegmentLayerRepairRatio("c0", 5, 5, 0.30),),
                certified_floor=0.15,
                profile_frozen=False,
                certified_ratio_candidates=(0.30,),
            )

    def test_development_measurement_scope_cannot_be_frozen(self):
        with self.assertRaisesRegex(ValueError, "cannot masquerade"):
            validate_union_repair_ratio_plan(
                scope=RepairRatioScope.DEVELOPMENT_PROFILE_MEASUREMENT,
                rows=(SegmentLayerRepairRatio("c0", 5, 5, 0.30),),
                certified_floor=0.15,
                profile_frozen=True,
                certified_ratio_candidates=(0.30,),
            )


if __name__ == "__main__":
    unittest.main()
