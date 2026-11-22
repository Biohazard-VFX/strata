import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from parity import overflow_field


class OverflowField(unittest.TestCase):
    def render(self, grid, size):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'field.json'
            path.write_text(json.dumps({'size': size, 'rgb8': grid}))
            return overflow_field(path)

    def test_corners_keep_grid_colours(self):
        grid = [[[0, 10, 20], [100, 110, 120]], [[200, 210, 220], [40, 50, 60]]]
        image = self.render(grid, [7, 9])
        self.assertEqual(image.shape, (7, 9, 3))
        self.assertEqual(image.dtype, np.uint8)
        for y, x, gy, gx in ((0, 0, 0, 0), (0, 8, 0, 1), (6, 0, 1, 0), (6, 8, 1, 1)):
            np.testing.assert_array_equal(image[y, x], grid[gy][gx])

    def test_interior_is_bilinear(self):
        image = self.render([[[0, 0, 0], [100, 0, 0]], [[0, 200, 0], [100, 200, 40]]], [3, 3])
        np.testing.assert_array_equal(image[1, 1], [50, 100, 10])
        np.testing.assert_array_equal(image[0, 1], [50, 0, 0])

    def test_committed_field_is_a_valid_image(self):
        image = overflow_field(Path(__file__).parent/'overflow_field.json')
        self.assertEqual(image.shape, (544, 960, 3))


if __name__ == '__main__':
    unittest.main()
