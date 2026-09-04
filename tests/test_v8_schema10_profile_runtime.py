import inspect
import unittest

from probekv.v6_a800_executor import RealCacheBlendA800Executor
from probekv.v8_schema10_profile_runtime import Schema10DevelopmentCaseRuntime


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


if __name__ == "__main__":
    unittest.main()
