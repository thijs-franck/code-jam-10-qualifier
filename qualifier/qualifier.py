import cv2
import numpy as np


def valid_input(image_size: tuple[int, int], tile_size: tuple[int, int], ordering: list[int]) -> bool:
    """
    Return True if the given input allows the rearrangement of the image, False otherwise.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once.
    """

    image_width, image_height = image_size
    tile_width, tile_height = tile_size

    remainder_width = image_width % tile_width
    remainder_height = image_height % tile_height

    n_tiles = (image_width / tile_width) * (image_height / tile_height)

    return not remainder_width and not remainder_height and len(set(ordering)) == n_tiles and min(ordering) == 0 and max(ordering) == n_tiles - 1


def rearrange_tiles(image_path: str, tile_size: tuple[int, int], ordering: list[int], out_path: str) -> None:
    original_image = cv2.imread(image_path)

    width, height, channels = original_image.shape
    tile_width, tile_height = tile_size

    tiles_per_column = height // tile_height
    tiles_per_row = width // tile_width
    rows, cols = np.divmod(ordering, tiles_per_row)

    # Split the image into an array of tiles
    tiled_array = (
        original_image
        .reshape(
            tiles_per_column,
            tile_height,
            tiles_per_row,
            tile_width,
            channels
        )
        .swapaxes(1, 2)
    )

    # Using advanced indexing to get rearranged tiles
    rearranged_tiles = tiled_array[rows, cols].reshape(
        tiles_per_column,
        tiles_per_row,
        tile_height,
        tile_width,
        channels,
    )

    # Reshape rearranged_tiles back into a regular image format
    rearranged_array = rearranged_tiles.swapaxes(1,2).reshape(height, width, channels)

    # Save the rearranged image
    cv2.imwrite(out_path, rearranged_array)
