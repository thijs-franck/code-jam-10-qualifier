from dataclasses import dataclass
import unittest
import unittest.mock

import numpy as np
from PIL import Image

from .qualifier import valid_input, rearrange_tiles

import os


@dataclass
class TestInfo:
    scrambled_image_path: str
    image_size: tuple[int, int]
    tile_size: tuple[int, int]
    ordering_path: str
    unscrambled_image_path: str

    def __post_init__(self):
        dirname = os.path.dirname(__file__)
        self.scrambled_image_path = os.path.join(dirname, self.scrambled_image_path)
        self.ordering_path = os.path.join(dirname, self.ordering_path)
        self.unscrambled_image_path = os.path.join(dirname, self.unscrambled_image_path)
        with open(self.ordering_path, 'r') as f:
            self.ordering = [int(x) for x in f.read().strip().splitlines()]


class ValidInputTest(unittest.TestCase):

    def setUp(self):
        self.images = [
            TestInfo("images/pydis_logo_scrambled.png", (512, 512), (256, 256), "images/pydis_logo_order.txt",
                     "images/pydis_logo_unscrambled.png"),
            TestInfo("images/great_wave_scrambled.png", (1104, 1600), (16, 16), "images/great_wave_order.txt",
                     "images/great_wave_unscrambled.png"),
            TestInfo("images/secret_image1_scrambled.png", (800, 600), (20, 20), "images/secret_image1_order.txt",
                     "images/secret_image1_unscrambled.png"),
            TestInfo("images/secret_image2_scrambled.png", (800, 600), (20, 20), "images/secret_image2_order.txt",
                     "images/secret_image2_unscrambled.png")
        ]

    def test_tile_size_doesnt_match_image_size(self):
        """Pass tile sizes that don't match the image size. The valid_input function should return False."""
        test_cases = [(63, 63), (65, 65), (1024, 1024)]
        for tile_size in test_cases:
            with self.subTest(tile_size=tile_size):
                self.assertFalse(valid_input(self.images[0].image_size, tile_size, self.images[0].ordering))

    def test_invalid_ordering(self):
        """Give an ordering that isn't a permutation of range(len(ordering))."""
        ordering = self.images[0].ordering.copy()
        ordering[-1] -= 1
        self.assertFalse(valid_input(self.images[0].image_size, (256, 256), ordering))

    def test_tile_size_doesnt_match_ordering(self):
        """Should not be valid if the length of `ordering` is not the number of tiles."""
        ordering = self.images[0].ordering.copy()
        ordering.append(len(ordering))  # Add this value specifically so the ordering itself remains valid.
        self.assertFalse(valid_input(self.images[0].image_size, (256, 256), ordering))

    def test_valid_input(self):
        """Make sure `valid_input` returns True for valid input."""
        test_cases = [
            ((1024, 1024), (128, 128), list(range(64))),
            ((1024, 1024), (256, 256), list(range(16))),
            ((1024, 1024), (512, 512), list(range(4))),
            ((1024, 1024), (1024, 1024), [0]),
            ((40, 60), (20, 20), list(range(6)))
        ]

        for image_size, tile_size, ordering in test_cases:
            with self.subTest(image_size=image_size, tile_size=tile_size, ordering=ordering):
                self.assertTrue(valid_input(image_size, tile_size, ordering))

    def test_invalid_input_raises_exception_in_reordering(self):

        with self.assertRaises(ValueError) as exc:
            rearrange_tiles(
                self.images[0].unscrambled_image_path, self.images[1].tile_size, self.images[2].ordering,
                "qualifier/images/user_output.png"
            )
        self.assertEqual("The tile size or ordering are not valid for the given image", str(exc.exception))

    def test_correct_ordering(self):
        """Run the images against the qualifier code, check if the unscrambled image is the proper one."""
        for image_index in range(len(self.images)):
            with self.subTest(image_index=image_index):
                rearrange_tiles(
                    self.images[image_index].scrambled_image_path,
                    self.images[image_index].tile_size,
                    self.images[image_index].ordering,
                    f"qualifier/images/user_output{image_index}.png"
                )

                correct_output = np.array(Image.open(self.images[image_index].unscrambled_image_path))
                user_output = np.array(Image.open(f"qualifier/images/user_output{image_index}.png"))

                self.assertTrue((user_output == correct_output).all())
    

if __name__ == "__main__":
    unittest.main()
