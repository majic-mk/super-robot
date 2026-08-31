from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

from .contracts import KVLocation
from .v7_contracts import CanonicalKVArtifact, PhysicalReplica
from .v8_leases import V8LeaseManager, V8ReplicaResource
from .v8_schema6_contracts import Gate2AxisState, PlannerSnapshot
from .v8_schema6_hbm import UnifiedHBMReservationManager
from .v8_schema6_runtime import Schema6RequestController


@dataclass
class Schema6TransferAuthorization:
    controller: Schema6RequestController
    segment_id: str
    lease_manager: V8LeaseManager
    source_variant_id: str
    artifact_id: str
    replica_id: str
    ready: bool = False
    released: bool = False

    def assert_valid_for(
        self, *, source_variant_id: str, artifact_id: str, replica_id: str
    ) -> None:
        if self.released:
            raise RuntimeError("full-KV transfer authorization was released")
        if (
            self.source_variant_id,
            self.artifact_id,
            self.replica_id,
        ) != (source_variant_id, artifact_id, replica_id):
            raise RuntimeError("full-KV transfer authorization binding differs")
        row = self.controller.records[self.segment_id]
        if not all(
            (row.logical_lease_id, row.physical_lease_id, row.hbm_reservation_id)
        ):
            raise RuntimeError("full-KV transfer authorization lost lease/HBM ownership")

    def mark_ready(self, *, actual_reuse_boundary: int) -> None:
        if self.released:
            raise RuntimeError("released transfer authorization cannot become ready")
        self.controller.mark_winner_ready(
            self.segment_id, actual_reuse_boundary=actual_reuse_boundary
        )
        self.ready = True

    def release(self, *, reason: str = "schema6_transfer_complete") -> None:
        if self.released:
            return
        row = self.controller.records[self.segment_id]
        if row.commit_state.value == "reuse_commit":
            self.controller.complete_reuse_execution(self.segment_id)
        else:
            self.controller.release_physical_preparation(self.segment_id, reason=reason)
            if row.logical_lease_id:
                self.lease_manager.release(row.logical_lease_id, reason=reason)
        self.released = True


class Schema6FullKVTransferAuthorizer:
    """Makes lease and HBM ownership a precondition of every full-KV copy."""

    def __init__(
        self,
        *,
        hbm_manager_provider: Callable[[], UnifiedHBMReservationManager],
    ) -> None:
        self.hbm_manager_provider = hbm_manager_provider
        self.lease_manager = V8LeaseManager()
        self._requests = itertools.count(1)
        self.authorized_transfers = 0
        self.nonwinner_transfers = 0
        self.transfers_before_source_freeze = 0

    def authorize(
        self,
        *,
        segment_id: str,
        source_variant_id: str,
        artifact: CanonicalKVArtifact,
        replica: PhysicalReplica,
        predicted_remaining_s: float,
    ) -> Schema6TransferAuthorization:
        if artifact.source_variant_id != source_variant_id:
            raise RuntimeError("full-KV authorization Source/Artifact mismatch")
        if replica.artifact_id != artifact.artifact_id:
            raise RuntimeError("full-KV authorization Artifact/Replica mismatch")
        source = self.lease_manager.register_source(
            source_variant_id, artifact.artifact_id, "schema6-sentinel"
        )
        if replica.replica_id not in source.replicas:
            self.lease_manager.register_replica(
                V8ReplicaResource(
                    replica.replica_id,
                    source_variant_id,
                    artifact.artifact_id,
                    KVLocation(replica.tier),
                    replica.generation,
                    replica.locator.placement_epoch,
                    replica.size_bytes,
                    True,
                )
            )
        request_number = next(self._requests)
        request_id = "schema6-transfer-%d" % request_number
        local_segment_id = "%s-r%d" % (segment_id, request_number)
        hbm = self.hbm_manager_provider()
        controller = Schema6RequestController(
            request_id=request_id,
            request_generation=1,
            ordered_segment_ids=(local_segment_id,),
            policy="immediate_staggered_closed_loop",
            lease_manager=self.lease_manager,
            hbm_manager=hbm,
        )
        controller.decision_ready(local_segment_id, source_variant_id, 1)
        controller.gate1(
            local_segment_id,
            passed=True,
            at_lmax=False,
            predicted_remaining_s=predicted_remaining_s,
        )
        snap = PlannerSnapshot(
            1, 1, "schema6-transfer-authorizer", hbm.epoch, "sparse-profile"
        )
        controller.apply_gate2(
            {local_segment_id: Gate2AxisState.PROVISIONAL_REUSE.value},
            snapshot=snap,
            current_snapshot=snap,
        )
        controller.begin_winner_prefetch(
            local_segment_id,
            artifact_id=artifact.artifact_id,
            replica_id=replica.replica_id,
            replica_generation=replica.generation,
            placement_epoch=replica.locator.placement_epoch,
            target_hbm_bytes=replica.size_bytes,
            predicted_remaining_s=predicted_remaining_s,
        )
        row = controller.records[local_segment_id]
        if not row.logical_lease_id:
            self.transfers_before_source_freeze += 1
            raise RuntimeError("full-KV authorization did not freeze the Source")
        if not row.physical_lease_id or not row.hbm_reservation_id:
            raise RuntimeError("full-KV authorization lacks lease or HBM reservation")
        self.authorized_transfers += 1
        return Schema6TransferAuthorization(
            controller,
            local_segment_id,
            self.lease_manager,
            source_variant_id,
            artifact.artifact_id,
            replica.replica_id,
        )

    def audit(self) -> dict[str, int | bool]:
        return {
            "authorized_full_kv_transfers": self.authorized_transfers,
            "full_kv_transfers_before_source_freeze": self.transfers_before_source_freeze,
            "nonwinner_full_kv_transfers": self.nonwinner_transfers,
            "passed": (
                self.transfers_before_source_freeze == 0
                and self.nonwinner_transfers == 0
            ),
        }
