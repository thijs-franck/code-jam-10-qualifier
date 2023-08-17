from PIL import Image


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
    """
    Rearrange the image.

    The image is given in `image_path`. Split it into tiles of size `tile_size`, and rearrange them by `ordering`.
    The new image needs to be saved under `out_path`.

    The tile size must divide each image dimension without remainders, and `ordering` must use each input tile exactly
    once. If these conditions do not hold, raise a ValueError with the message:
    "The tile size of ordering are not valid for the given image".
    """

    with (
        Image.open(image_path) as original_image,
        Image.new(original_image.mode, original_image.size) as reordered_image
    ):

        if not valid_input(original_image.size, tile_size, ordering):
            raise ValueError(
                "The tile size or ordering are not valid for the given image")

        tile_width, tile_height = tile_size
        tiles_per_row = original_image.width // tile_width

        for position, index in enumerate(ordering):
            # Find the top left corner of the segment in the original image
            image_x = (index % tiles_per_row) * tile_width
            image_y = (index // tiles_per_row) * tile_height

            # Calculate the top left corner in the reordered image where the segment should be pasted
            out_x = (position % tiles_per_row) * tile_width
            out_y = (position // tiles_per_row) * tile_height

            # Crop the segment out of the original image and paste it into the reordered image
            segment = original_image.crop((
                image_x,
                image_y,
                image_x + tile_width,
                image_y + tile_height
            ))

            reordered_image.paste(segment, (out_x, out_y))
        # END LOOP

        reordered_image.save(out_path)
