import copy
import importlib.util
from pathlib import Path
import unittest

from probekv.v8_schema10_execution import digest_json

spec = importlib.util.spec_from_file_location("capture_cli", Path(__file__).resolve().parents[1]/"scripts/server/run_source_policy_capture.py")
capture_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture_cli)


def inputs():
    def request(name, epoch, head):
        return dict(request_id=name,request_epoch=epoch,token_ids=[head,4,5,6,9],
            segments=[dict(segment_id="C",positions=[1,2,3],token_ids=[4,5,6],content_key="exact")],
            mandatory_suffix_positions=[4],partition_role="development",
            development_partition_digest="b"*64,locked_test_accessed=False)
    sources = [request("old1",1,10),request("old2",2,20)]
    target = request("now",3,30)
    plan = dict(kind="source_policy_native_capture_plan_v1",source_requests=sources,target_request=target,
                selection_path="d1_d2_rescue",host_budget_bytes=1000000,global_byte_budget=1000000,
                paper_evidence=False,locked_test_accessed=False)
    plan["plan_sha256"] = digest_json(plan)
    partition = dict(kind="source_policy_capture_partition_v1",locked_test_accessed=False,
        parent_development_partition_sha256="b"*64,
        entries=[dict(request_id=q["request_id"],request_sha256=digest_json(q),role="development",content_group="same-content") for q in sources+[target]])
    return plan,partition


class CapturePlanTests(unittest.TestCase):
    def test_valid_plan_requires_no_model_or_gpu(self):
        capture_cli.validate_plan(*inputs())

    def test_changed_source_input_requires_new_manifest_and_partition(self):
        plan,part = inputs()
        plan["source_requests"][0]["token_ids"][0] = 99
        with self.assertRaisesRegex(ValueError,"immutable"):
            capture_cli.validate_plan(plan,part)
        plan["plan_sha256"] = digest_json({k:v for k,v in plan.items() if k!="plan_sha256"})
        with self.assertRaisesRegex(ValueError,"exact member"):
            capture_cli.validate_plan(plan,part)

    def test_group_leakage_rejected(self):
        plan,part = inputs()
        part["entries"][0]["role"] = "fit"
        part["entries"][1]["role"] = "validation"
        with self.assertRaisesRegex(ValueError,"leaks"):
            capture_cli.validate_plan(plan,part)

    def test_future_source_is_not_visible(self):
        plan,part = inputs()
        plan["source_requests"][0]["request_epoch"] = 4
        part["entries"][0]["request_sha256"] = digest_json(plan["source_requests"][0])
        plan["plan_sha256"] = digest_json({k:v for k,v in plan.items() if k!="plan_sha256"})
        with self.assertRaisesRegex(ValueError,"earlier exact"):
            capture_cli.validate_plan(plan,part)

    def test_mandatory_suffix_cannot_be_repair_segment(self):
        plan,part = inputs()
        q = plan["target_request"]
        q["segments"][0].update(positions=[2,3,4],token_ids=[5,6,9])
        part["entries"][-1]["request_sha256"] = digest_json(q)
        plan["plan_sha256"] = digest_json({k:v for k,v in plan.items() if k!="plan_sha256"})
        with self.assertRaisesRegex(ValueError,"suffix"):
            capture_cli.validate_plan(plan,part)


if __name__ == "__main__":
    unittest.main()
