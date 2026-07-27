import random
import unittest

from probekv.rope import apply_rope, relative_l2_error, rope_angles


class RopeTests(unittest.TestCase):
    def test_derotate_rerotate_round_trip(self):
        randomizer = random.Random(20260726)
        vector = [randomizer.uniform(-2, 2) for _ in range(128)]
        cosine, sine = rope_angles(position=1234, head_dim=128)
        rotated = apply_rope(vector, cosine, sine)
        recovered = apply_rope(rotated, cosine, sine, inverse=True)
        self.assertLessEqual(relative_l2_error(vector, recovered), 1e-12)

    def test_invalid_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            rope_angles(1, 127)


if __name__ == "__main__":
    unittest.main()
