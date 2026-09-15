import unittest
from .preview import byte_range


class PreviewRangeTests(unittest.TestCase):
    def test_whole(self):
        self.assertEqual(byte_range(None, 10), (0, 9, 200))
        self.assertEqual(byte_range(None, 0), (0, -1, 200))

    def test_closed_open_and_suffix(self):
        for header, expected in [('bytes=2-4', (2, 4, 206)), ('bytes=2-', (2, 9, 206)),
                                 ('bytes=-3', (7, 9, 206)), ('bytes=8-99', (8, 9, 206)),
                                 ('bytes=-99', (0, 9, 206))]:
            self.assertEqual(byte_range(header, 10), expected)

    def test_invalid(self):
        for header in ['bytes=10-', 'bytes=7-4', 'bytes=-0', 'bytes=-', 'bytes=x-2',
                       'bytes=0-1,3-4', 'items=1-2']:
            with self.subTest(header=header), self.assertRaises(ValueError):
                byte_range(header, 10)


if __name__ == '__main__':
    unittest.main()
