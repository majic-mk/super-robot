import unittest

from scripts.server.verify_paper_environment import evaluate_environment


def contract():
    return {
        "hardware": {
            "primary": {
                "gpu_name_regex": r"^NVIDIA A800.*80GB$",
                "gpu_count": 1,
                "minimum_memory_mib": 80000,
                "compute_capability": "8.0",
            }
        },
        "stacks": {
            "primary": {
                "pytorch": "2.2.1",
                "vllm": "0.4.1",
                "xformers": "0.0.25",
                "cuda": "12.1",
            }
        },
    }


def valid_record():
    return {
        "gpus": [
            {
                "name": "NVIDIA A800-SXM4-80GB",
                "memory_mib": 81920,
                "uuid": "GPU-fixture",
                "compute_capability": "8.0",
            }
        ],
        "torch_device_count": 1,
        "torch": "2.2.1",
        "vllm": "0.4.1",
        "xformers": "0.0.25",
        "torch_cuda": "12.1",
        "nvcc_cuda": "12.1",
        "git_status": "",
    }


class PaperEnvironmentTests(unittest.TestCase):
    def test_a800_environment_passes(self):
        self.assertEqual(evaluate_environment(contract(), valid_record()), [])

    def test_a100_cannot_be_reported_as_a800(self):
        record = valid_record()
        record["gpus"][0]["name"] = "NVIDIA A100-SXM4-80GB"
        failures = evaluate_environment(contract(), record)
        self.assertTrue(any("GPU name" in failure for failure in failures))

    def test_dirty_or_wrong_stack_fails(self):
        record = valid_record()
        record["torch"] = "2.1.2"
        record["git_status"] = " M configs/experiment_contract.yaml"
        failures = evaluate_environment(contract(), record)
        self.assertTrue(any("torch==2.2.1" in failure for failure in failures))
        self.assertTrue(any("clean worktree" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
