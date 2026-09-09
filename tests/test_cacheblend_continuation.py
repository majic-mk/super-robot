import unittest
from types import SimpleNamespace

from probekv.cacheblend_continuation import DenseLoopContinuation


class TensorStub:
    def __init__(self, shape=(4, 8)):
        self.shape, self.dtype, self.device = shape, "bf16", "cuda:0"

    def __getitem__(self, key):
        return self


class ContinuationTests(unittest.TestCase):
    def fixture(self, depth=1):
        calls = []
        def layer(index):
            def forward(pos, h, kv, attn, r, **kw):
                calls.append((index + 1, kw["status"]))
                kw["cache_fuse_metadata"]["imp_indices"] = [0, 1, 2, 3]
                return h, r
            return forward
        model = SimpleNamespace(layers=[layer(i) for i in range(4)], old_kvs=[None] * 4,
            norm=lambda h, r: (h, r), cache_fuse_metadata=dict(check=True,
                check_layers=[depth], probekv_resumable=False,
                probekv_matched_boundary_source_kv=True))
        session = SimpleNamespace(token_ids=(1, 2, 3, 4), _started=True, _finished=False,
            exact_prefix_tokens=0, commits={}, _pending_target_positions=None,
            current_layer=depth, active_positions=(0, 1, 2, 3),
            hidden_states=TensorStub(), residual=TensorStub(),
            attention_metadata=object(), working_kv=[None] * 4)
        pos = TensorStub((4,))
        args = dict(request_id="q", generation=depth, inner_model=model,
                    boundary=depth + 1, positions=pos)
        return session, model, pos, calls, args

    def test_move_once_and_skip_completed_layers_d1_d2(self):
        for depth in (1, 2):
            s, m, p, calls, args = self.fixture(depth)
            original = s.hidden_states
            c = DenseLoopContinuation.detach(s, **args)
            self.assertTrue(s._finished)
            self.assertIsNone(s.hidden_states)
            with self.assertRaises(ValueError):
                DenseLoopContinuation.detach(s, **args)
            kwargs = dict(request_id="q", generation=depth, positions=p,
                          attention_metadata=s.attention_metadata, working_kv=s.working_kv)
            self.assertIs(c.run(**kwargs), original)
            self.assertEqual(calls, [(i, 1 if i == depth + 1 else 2) for i in range(depth + 1, 5)])
            self.assertEqual(c.executed_layers, list(range(depth + 1, 5)))
            with self.assertRaises(RuntimeError):
                c.run(**kwargs)

    def test_prefix_sparse_committed_and_wrong_boundary_rejected(self):
        for change in (dict(exact_prefix_tokens=1), dict(commits={"C": 2}),
                       dict(active_positions=(1, 2, 3)), dict(current_layer=0),
                       dict(_pending_target_positions=(1, 2)), dict(residual=None)):
            s, m, p, calls, args = self.fixture()
            s.__dict__.update(change)
            with self.assertRaises(ValueError):
                DenseLoopContinuation.detach(s, **args)
            self.assertFalse(s._finished)

    def test_stale_request_generation_kv_attention_and_positions(self):
        for key, value in (("request_id", "other"), ("generation", 99),
                           ("working_kv", [None] * 4), ("attention_metadata", object()),
                           ("positions", TensorStub((4,)))):
            s, m, p, calls, args = self.fixture()
            c = DenseLoopContinuation.detach(s, **args)
            kwargs = dict(request_id="q", generation=1, positions=p,
                          attention_metadata=s.attention_metadata, working_kv=s.working_kv)
            kwargs[key] = value
            with self.assertRaises(RuntimeError):
                c.run(**kwargs)
            self.assertEqual(calls, [])

    def test_exception_consumes_state_and_does_not_retry(self):
        s, m, p, calls, args = self.fixture()
        c = DenseLoopContinuation.detach(s, **args)
        def fail(*args, **kwargs):
            raise RuntimeError("kernel failure")
        m.layers[1] = fail
        kwargs = dict(request_id="q", generation=1, positions=p,
                      attention_metadata=s.attention_metadata, working_kv=s.working_kv)
        with self.assertRaisesRegex(RuntimeError, "kernel failure"):
            c.run(**kwargs)
        self.assertTrue(c.consumed)
        self.assertIsNone(c.hidden_states)
        with self.assertRaisesRegex(RuntimeError, "replayed"):
            c.run(**kwargs)

    def test_resumable_or_unmatched_metadata_cannot_cross_handoff(self):
        for change in (dict(probekv_resumable=True), dict(check_layers=[2]),
                       dict(probekv_matched_boundary_source_kv=False)):
            s, m, p, calls, args = self.fixture()
            c = DenseLoopContinuation.detach(s, **args)
            m.cache_fuse_metadata.update(change)
            with self.assertRaises(RuntimeError):
                c.run(request_id="q", generation=1, positions=p,
                      attention_metadata=s.attention_metadata, working_kv=s.working_kv)
            self.assertEqual(calls, [])
