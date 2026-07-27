import unittest

from probekv.reference_hf import HuggingFaceReferenceStateBackend


class FakeModel:
    def eval(self):
        return self


class StableTokenizer:
    def encode(self, text, add_special_tokens):
        values = [ord(character) for character in text]
        return ([0] + values) if add_special_tokens else values


class BoundaryMergingTokenizer(StableTokenizer):
    def encode(self, text, add_special_tokens):
        if text == "A B":
            return [0, 999]
        return super().encode(text, add_special_tokens)


class ReferenceBackendTests(unittest.TestCase):
    def test_boundary_stable_parts_are_accepted(self):
        backend = HuggingFaceReferenceStateBackend(FakeModel(), StableTokenizer())
        prefix, segment = backend.tokenize_parts("A ", "B")
        self.assertEqual(prefix + segment, (0, 65, 32, 66))

    def test_boundary_merge_is_rejected(self):
        backend = HuggingFaceReferenceStateBackend(
            FakeModel(), BoundaryMergingTokenizer()
        )
        with self.assertRaisesRegex(ValueError, "boundary-stable"):
            backend.tokenize_parts("A ", "B")


if __name__ == "__main__":
    unittest.main()
