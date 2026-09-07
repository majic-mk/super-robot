"""vLLM 0.4.1 block-manager ownership, never diagnostic fixed block ranges.

API source: vllm/core/block_manager_v1.py and vllm/sequence.py at v0.4.1.
This allocation primitive does NOT by itself certify a complete model adapter.
"""
from __future__ import annotations

from itertools import count
import time


class NativeBlockRequest:
    _sequence_ids = count(10_000_000)

    def __init__(self, *, block_manager, request_id, prompt_token_ids, sampling_params,
                 synchronize, apply_block_copies, prefix_shadow_provider, types=None):
        if types is None:
            from vllm.sequence import Sequence, SequenceGroup, SequenceGroupMetadata, SequenceStatus, Logprob
            from vllm.core.interfaces import AllocStatus
            types = dict(Sequence=Sequence, SequenceGroup=SequenceGroup,
                         Metadata=SequenceGroupMetadata, Status=SequenceStatus, Logprob=Logprob, AllocStatus=AllocStatus)
        if not block_manager.enable_caching:
            raise RuntimeError("native exact Prefix Cache must be enabled")
        self.manager, self.types, self.synchronize = block_manager, types, synchronize
        self.apply_block_copies, self.shadow_provider = apply_block_copies, prefix_shadow_provider
        self.sequence = types["Sequence"](seq_id=next(self._sequence_ids), prompt="",
            prompt_token_ids=list(prompt_token_ids), block_size=block_manager.block_size)
        if self.sequence.seq_id in block_manager.block_tables:
            raise RuntimeError("native sequence ID collision")
        self.group = types["SequenceGroup"](request_id=request_id, seqs=[self.sequence],
            sampling_params=sampling_params, arrival_time=time.monotonic())
        self.allocated = self.closed = self.prefill_finished = False
        self.cached_block_ids = ()
        self.prefix_shadow = None

    def __enter__(self):
        if self.manager.can_allocate(self.group) != self.types["AllocStatus"].OK:
            raise MemoryError("native allocator cannot admit this request")
        try:
            self.manager.allocate(self.group)
            self.allocated = True
            self.cached_block_ids = tuple(self.manager.get_common_computed_block_ids([self.sequence]))
            table = tuple(self.manager.get_block_table(self.sequence))
            if table[:len(self.cached_block_ids)] != self.cached_block_ids:
                raise RuntimeError("computed native blocks are not an exact prefix")
            if self.cached_prefix_tokens:
                self.prefix_shadow = self.shadow_provider(self.sequence.get_prompt_token_ids()[:self.cached_prefix_tokens],
                                                          self.cached_block_ids)
                if self.prefix_shadow is None:
                    raise RuntimeError("native Prefix hit lacks verified pre-RoPE shadow")
            self.sequence.status = self.types["Status"].RUNNING
            self.manager.access_all_blocks_in_seq(self.sequence, time.monotonic())
            return self
        except Exception:
            self.close()
            raise

    @property
    def cached_prefix_tokens(self):
        return len(self.cached_block_ids) * self.manager.block_size

    def metadata(self, *, is_prompt):
        if not self.allocated or self.closed:
            raise RuntimeError("request has no native block lease")
        seq = self.sequence
        return self.types["Metadata"](request_id=self.group.request_id, is_prompt=is_prompt,
            seq_data={seq.seq_id: seq.data}, sampling_params=self.group.sampling_params,
            block_tables={seq.seq_id: self.manager.get_block_table(seq)},
            computed_block_nums=list(self.cached_block_ids) if is_prompt else [],
            token_chunk_size=seq.get_prompt_len() if is_prompt else 1)

    def finish_prefill(self, *, exact_dense: bool):
        if self.prefill_finished or self.closed:
            raise RuntimeError("prefill completion is not repeatable")
        self.synchronize()
        # Selective KV must NEVER poison the engine's exact Prefix namespace.
        if exact_dense:
            self.manager.mark_blocks_as_computed(self.group)
        self.sequence.data.update_num_computed_tokens(self.sequence.get_prompt_len())
        self.prefill_finished = True

    def append_for_decode(self, token_id):
        if not self.prefill_finished or self.closed:
            raise RuntimeError("decode requires completed owned prefill")
        if not self.manager.can_append_slots(self.group, 0):
            raise MemoryError("native allocator cannot append a decode slot")
        self.sequence.append_token_id(int(token_id), {int(token_id): self.types["Logprob"](logprob=0.0)})
        copies = self.manager.append_slots(self.sequence, 0)
        if copies:
            self.apply_block_copies(copies)
            self.synchronize()
        return self.metadata(is_prompt=False)

    def finish_decode_step(self):
        self.sequence.data.update_num_computed_tokens(1)

    def close(self):
        if not self.closed:
            self.synchronize()
            # free() is idempotent and handles failure before full allocation.
            self.manager.free(self.sequence)
            self.closed = True

    def __exit__(self, *args):
        self.close()
