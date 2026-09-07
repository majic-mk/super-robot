from types import SimpleNamespace as NS
import unittest

from probekv.v8_schema10_native_blocks import NativeBlockRequest


class NativeBlocks(unittest.TestCase):
    def lease(self, *, shadow=True, allocate=True):
        class Sequence:
            def __init__(self, seq_id, prompt, prompt_token_ids, block_size):
                self.seq_id = seq_id
                self.tokens = prompt_token_ids
                self.data = NS(update_num_computed_tokens=lambda count: None)
            def get_prompt_token_ids(self): return self.tokens
            def get_prompt_len(self): return len(self.tokens)
            def append_token_id(self, token_id, logprobs): self.tokens.append(token_id)
        class Manager:
            enable_caching, block_size = True, 16
            def __init__(self): self.block_tables, self.marked, self.freed = {}, False, False
            def can_allocate(self, group): return "ok" if allocate else "later"
            def allocate(self, group): self.block_tables[group.seqs[0].seq_id] = [41, 9, 73]
            def get_common_computed_block_ids(self, seqs): return [41]
            def get_block_table(self, seq): return self.block_tables[seq.seq_id]
            def access_all_blocks_in_seq(self, *args): pass
            def mark_blocks_as_computed(self, *args): self.marked = True
            def can_append_slots(self, *args): return True
            def append_slots(self, *args): return [(73, 82)]
            def free(self, seq): self.freed = True; self.block_tables.pop(seq.seq_id, None)
        manager = Manager()
        calls = []
        types = {"Sequence": Sequence, "SequenceGroup": lambda **kw: NS(**kw),
                 "Metadata": lambda **kw: NS(**kw), "Status": NS(RUNNING="running"),
                 "Logprob": lambda **kw: NS(**kw), "AllocStatus": NS(OK="ok")}
        lease = NativeBlockRequest(block_manager=manager, request_id="q", prompt_token_ids=list(range(33)),
            sampling_params=NS(), synchronize=lambda: calls.append("fence"),
            apply_block_copies=lambda value: calls.append(value),
            prefix_shadow_provider=lambda tokens, blocks: {"tokens": tokens, "blocks": blocks} if shadow else None,
            types=types)
        return manager, lease, calls

    def test_block_ids_come_from_native_manager_and_prefix_metadata(self):
        manager, lease, calls = self.lease()
        with lease:
            metadata = lease.metadata(is_prompt=True)
            self.assertEqual(list(metadata.block_tables.values()), [[41, 9, 73]])
            self.assertEqual(metadata.computed_block_nums, [41])
            self.assertEqual(lease.cached_prefix_tokens, 16)
            lease.finish_prefill(exact_dense=True)
            lease.append_for_decode(8)
        self.assertTrue(manager.marked)
        self.assertTrue(manager.freed)
        self.assertIn([(73, 82)], calls)

    def test_selective_prefill_cannot_publish_exact_prefix(self):
        manager, lease, _ = self.lease()
        with lease:
            lease.finish_prefill(exact_dense=False)
        self.assertFalse(manager.marked)

    def test_missing_prefix_shadow_fails_and_releases_native_blocks(self):
        manager, lease, _ = self.lease(shadow=False)
        with self.assertRaises(RuntimeError):
            with lease:
                pass
        self.assertTrue(manager.freed)

    def test_no_fixed_range_fallback_when_allocator_cannot_admit(self):
        manager, lease, _ = self.lease(allocate=False)
        with self.assertRaises(MemoryError):
            with lease:
                pass
        self.assertEqual(manager.block_tables, {})

if __name__ == "__main__": unittest.main()
