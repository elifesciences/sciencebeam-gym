from cv2 import cv2 as cv
import numpy as np

from sciencebeam_gym.utils.bounding_box import BoundingBox


def resize_image(src: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv.resize(
        src,
        dsize=(width, height),
        interpolation=cv.INTER_CUBIC
    )


def copy_image_to(
    src: np.ndarray,
    dst: np.ndarray,
    dst_bounding_box: BoundingBox,
):
    x = int(dst_bounding_box.x)
    y = int(dst_bounding_box.y)
    width = int(dst_bounding_box.width)
    height = int(dst_bounding_box.height)
    dst[y:(y + height), x:(x + width)] = resize_image(
        src, width=width, height=height
    )
