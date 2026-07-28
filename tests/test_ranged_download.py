import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.server.assemble_ranged_download import assemble, git_blob_sha1
from scripts.server.download_http_range import parse_content_range


class RangedDownloadTests(unittest.TestCase):
    def test_content_range_parser(self):
        self.assertEqual(
            parse_content_range("bytes 10-19/100"),
            (10, 19, 100),
        )
        self.assertEqual(
            parse_content_range("bytes 10-19/*"),
            (10, 19, None),
        )
        with self.assertRaises(ValueError):
            parse_content_range("10-19/100")

    def test_assembly_requires_exact_part_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = []
            for index, payload in enumerate((b"abcd", b"efgh", b"ij")):
                path = root / ("part%02d" % index)
                path.write_bytes(payload)
                parts.append(path)
            output = root / "complete.bin"
            assemble(parts, output, total_bytes=10, regular_part_bytes=4)
            self.assertEqual(output.read_bytes(), b"abcdefghij")
            expected = hashlib.sha1(b"blob 10\0abcdefghij").hexdigest()
            self.assertEqual(git_blob_sha1(output), expected)

    def test_assembly_rejects_incomplete_part(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = []
            for index, payload in enumerate((b"abcd", b"efg", b"ij")):
                path = root / ("part%02d" % index)
                path.write_bytes(payload)
                parts.append(path)
            with self.assertRaises(ValueError):
                assemble(
                    parts,
                    root / "complete.bin",
                    total_bytes=10,
                    regular_part_bytes=4,
                )


if __name__ == "__main__":
    unittest.main()
