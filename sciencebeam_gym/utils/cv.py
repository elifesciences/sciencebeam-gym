import logging

from cv2 import cv2 as cv
import numpy as np

from sciencebeam_gym.utils.bounding_box import BoundingBox


LOGGER = logging.getLogger(__name__)


def resize_image(src: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv.resize(
        src,
        dsize=(width, height),
        interpolation=cv.INTER_CUBIC
    )


def crop_image_to_bounding_box(
    src: np.ndarray,
    bounding_box: BoundingBox,
) -> np.ndarray:
    x = int(bounding_box.x)
    y = int(bounding_box.y)
    width = int(bounding_box.width)
    height = int(bounding_box.height)
    return src[x:(x + width), y:(y + height)]


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
