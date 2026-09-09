import unittest
from dataclasses import replace, FrozenInstanceError

from probekv.repair_backend_contract import ResidentRepairPlan
from probekv.v8_schema10_execution import digest_json


class ResidentRepairContractTests(unittest.TestCase):
    def plan(self, **changes):
        p = ResidentRepairPlan("source-A", "digest-A", digest_json(list(range(832))),
            2, .15, tuple(range(288, 800)), tuple(range(288, 365)), 832, 32)
        return replace(p, **changes)

    def test_ceiling_and_mandatory_rows(self):
        p = self.plan()
        self.assertEqual(len(p.repair_positions), 77)
        self.assertEqual(len(p.active_positions), 397)
        self.assertTrue(set(range(288)) <= set(p.active_positions))
        self.assertTrue(set(range(800, 832)) <= set(p.active_positions))

    def test_r1_owns_all_rows(self):
        p = self.plan(ratio=1., repair_positions=tuple(range(288, 800)))
        self.assertEqual(p.active_positions, tuple(range(832)))

    def test_bad_masks_rejected(self):
        for mask in (tuple(range(77)), tuple(range(288, 364)), (288,) * 77,
                     tuple(reversed(range(288, 365))), tuple(range(799, 876))):
            with self.subTest(mask=mask[:3]), self.assertRaises(ValueError):
                self.plan(repair_positions=mask)

    def test_prefix_not_silently_disabled(self):
        with self.assertRaises(ValueError):
            self.plan(cached_prefix_tokens=256)

    def test_identity_and_depth_binding(self):
        p = self.plan()
        binding = dict(source_id=p.source_id, token_ids=list(range(832)),
            positions=p.segment_positions, boundary=2, ratio=.15, cached_prefix_tokens=0)
        p.assert_binding(**binding)
        for change in (dict(source_id="B"), dict(boundary=3), dict(ratio=1.),
                       dict(cached_prefix_tokens=256), dict(token_ids=list(range(831)))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.assert_binding(**{**binding, **change})

    def test_execution_mask_must_match_exactly(self):
        p = self.plan()
        p.assert_execution(p.repair_positions, p.active_positions)
        with self.assertRaises(RuntimeError):
            p.assert_execution(p.repair_positions, p.active_positions[:-1])
        with self.assertRaises(RuntimeError):
            p.assert_execution(tuple(range(289, 366)), p.active_positions)

    def test_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan().source_id = "B"
        with self.assertRaises(ValueError):
            self.plan(repair_positions=list(range(288, 365)))

    def test_ratio_invalid(self):
        for ratio in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                self.plan(ratio=ratio)


if __name__ == "__main__":
    unittest.main()
