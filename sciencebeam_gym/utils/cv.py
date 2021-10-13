import logging
from typing import Optional

import PIL.Image
from cv2 import cv2 as cv
import numpy as np

from sciencebeam_gym.utils.bounding_box import BoundingBox


LOGGER = logging.getLogger(__name__)


def to_opencv_image(pil_image: PIL.Image.Image) -> np.ndarray:
    return cv.cvtColor(np.array(pil_image.convert('RGB')), cv.COLOR_RGB2BGR)


def get_pil_image_for__opencv_image(opencv_image: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(
        cv.cvtColor(opencv_image, cv.COLOR_BGR2RGB)
    )


def resize_image(
    src: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    assert width or height
    if not height:
        height = int(width * src.shape[0] / src.shape[1])
    if not width:
        width = int(height * src.shape[1] / src.shape[0])
    return cv.resize(
        src,
        dsize=(width, height),
        interpolation=cv.INTER_CUBIC
    )


def get_image_array_with_max_resolution(
    image_array: np.ndarray,
    max_width: int,
    max_height: int
) -> np.ndarray:
    original_height, original_width = image_array.shape[:2]
    if (
        (not max_width or original_width <= max_width)
        and (not max_height or original_height <= max_height)
    ):
        LOGGER.debug(
            'image within expected dimension: %sx%s <= %sx%s',
            original_width, original_height, max_width, max_height
        )
        return image_array
    target_width_based_on_height = int(
        original_width * max_height / original_height
    )
    target_height_based_on_width = int(
        original_height * max_width / original_width
    )
    if max_height and (not max_width or target_width_based_on_height <= max_width):
        return resize_image(
            image_array, width=target_width_based_on_height, height=max_height
        )
    return resize_image(
        image_array, width=max_width, height=target_height_based_on_width
    )


def load_pil_image_from_file(image_path: str) -> PIL.Image.Image:
    return get_pil_image_for__opencv_image(
        cv.imread(image_path)
    )


def crop_image_to_bounding_box(
    src: np.ndarray,
    bounding_box: BoundingBox,
) -> np.ndarray:
    x = int(bounding_box.x)
    y = int(bounding_box.y)
    width = int(bounding_box.width)
    height = int(bounding_box.height)
    return src[y:(y + height), x:(x + width)]


def copy_image_to(
    src: np.ndarray,
    dst: np.ndarray,
    dst_bounding_box: BoundingBox,
):
    x = int(dst_bounding_box.x)
    y = int(dst_bounding_box.y)
    width = int(dst_bounding_box.width)
    height = int(dst_bounding_box.height)
    resized_image = resize_image(
        src, width=width, height=height
    )
    LOGGER.debug(
        'dst_bounding_box: %s (resized_image.shape: %s, dst.shape: %s)',
        dst_bounding_box, resized_image.shape, dst.shape
    )
    dst[y:(y + height), x:(x + width)] = resized_image
