import unittest

from probekv.cacheblend_v6_online_engine import integrity_mode_performs_full_digest
from probekv.v8_schema7_contracts import (
    IntegrityAudit,
    IntegrityVerificationMode,
    TransferPath,
)
from probekv.v8_schema7_transfer import (
    PinnedStagingPool,
    Schema7TransferPlanner,
    TransferCapabilities,
)


class Schema7TransferIntegrityTests(unittest.TestCase):
    def test_online_path_never_performs_full_digest(self):
        self.assertFalse(integrity_mode_performs_full_digest("online_immutable"))
        self.assertFalse(integrity_mode_performs_full_digest("online_sampled"))
        self.assertTrue(integrity_mode_performs_full_digest("qualification_full"))
        IntegrityAudit(
            IntegrityVerificationMode.ONLINE_IMMUTABLE,
            "artifact", "", "", "", False,
        )

    def test_qualification_requires_destination_and_source_equality(self):
        IntegrityAudit(
            IntegrityVerificationMode.QUALIFICATION_FULL,
            "same", "same", "same", "same", True,
        )
        with self.assertRaises(RuntimeError):
            IntegrityAudit(
                IntegrityVerificationMode.QUALIFICATION_FULL,
                "same", "same", "same", "corrupt", True,
            )

    def test_gds_capability_falls_back_to_staged(self):
        planner = Schema7TransferPlanner()
        staged = planner.choose(
            source_tier="ssd", requested_bytes=1024,
            capabilities=TransferCapabilities(gds_available=False),
        )
        self.assertEqual(staged.path, TransferPath.SSD_STAGED_TO_GPU)
        direct = planner.choose(
            source_tier="ssd", requested_bytes=1024,
            capabilities=TransferCapabilities(gds_available=True),
        )
        self.assertEqual(direct.path, TransferPath.SSD_GDS_TO_GPU)

    def test_pinned_pool_double_buffer_is_byte_accounted(self):
        pool = PinnedStagingPool(100)
        lease = pool.acquire(owner_request_id="r", slot_bytes=40, double_buffer=True)
        self.assertEqual(pool.reserved_bytes, 80)
        with self.assertRaises(MemoryError):
            pool.acquire(owner_request_id="r2", slot_bytes=20, double_buffer=True)
        pool.release(lease.lease_id)
        self.assertEqual(pool.reserved_bytes, 0)


if __name__ == "__main__":
    unittest.main()
