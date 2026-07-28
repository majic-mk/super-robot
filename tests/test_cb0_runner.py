import unittest

from scripts.server.run_cb0_patched import detect_forbidden_errors


class CB0RunnerTests(unittest.TestCase):
    def test_vllm_memory_advice_is_not_an_oom(self):
        log = (
            "CUDA graphs can take additional memory. If you are running out "
            "of memory, consider decreasing gpu_memory_utilization."
        )
        self.assertEqual(detect_forbidden_errors(log), [])

    def test_real_runtime_failures_are_detected(self):
        log = (
            "Traceback (most recent call last):\n"
            "torch.cuda.OutOfMemoryError: CUDA out of memory."
        )
        detected = detect_forbidden_errors(log)
        self.assertIn("Traceback (most recent call last)", detected)
        self.assertIn("torch.cuda.OutOfMemoryError", detected)
        self.assertIn("CUDA out of memory", detected)


if __name__ == "__main__":
    unittest.main()
