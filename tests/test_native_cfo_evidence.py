import unittest

from probekv.v8_schema10_native_validation import validate_correctness_observation


class NativeCFOEvidenceTests(unittest.TestCase):
    def row(self):
        return dict(origin='real_cuda_execution', fake_timing=False, eager_reference=True,
                    expected_layers=32, captured_layers=32, eager_layer_errors=[1e-6] * 32,
                    eager_tolerance=2e-5, metadata_digest='actual-metadata',
                    post_rope_qk=True, causal_mask=True, gqa_mapping=True,
                    fp32_accumulation=True, streaming_logsumexp=True)

    def test_every_layer_required_and_pass_flag_cannot_replace_errors(self):
        self.assertTrue(validate_correctness_observation('cfo', self.row()))
        for changes in (
            dict(eager_layer_errors=[]), dict(eager_reference=False),
            dict(eager_layer_errors=[1e-6] * 31), dict(eager_layer_errors=[float('nan')] * 32),
            dict(eager_layer_errors=[3e-5] * 32), dict(eager_tolerance=1e-3),
            dict(gqa_mapping=False), dict(fake_timing=True),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_correctness_observation('cfo', {**self.row(), **changes, 'passed': True})
