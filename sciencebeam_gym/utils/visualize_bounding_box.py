import logging
from typing import Tuple

import numpy as np
from cv2 import cv2 as cv

from sciencebeam_gym.utils.bounding_box import BoundingBox


LOGGER = logging.getLogger(__name__)


def draw_bounding_box(
    image_array: np.ndarray,
    bounding_box: BoundingBox,
    color: Tuple[int, int, int],
    text: str,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    font_thickness: int = 2,
    text_offset_x: int = 2,
    text_offset_y: int = -1,
    text_margin: int = 5,
    text_color: Tuple[int, int, int] = (255, 255, 255)
):
    LOGGER.debug(
        'bounding_box: %s (color: %s [%s])',
        bounding_box, color, type(color[0])
    )
    cv.rectangle(
        image_array,
        (bounding_box.x, bounding_box.y),
        (
            bounding_box.x + bounding_box.width - 1,
            bounding_box.y + bounding_box.height - 1
        ),
        color,
        3
    )
    font = cv.FONT_HERSHEY_SIMPLEX
    LOGGER.debug('font: %r (%s)', font, type(font))
    (text_width, text_height), baseline = cv.getTextSize(
        text, font, font_scale, font_thickness
    )
    LOGGER.debug('text size: %s x %s, %s', text_width, text_height, baseline)
    text_x = bounding_box.x + text_offset_x + text_margin
    text_margin_bottom = max(text_margin, baseline)
    if text_offset_y < 0:
        text_y = bounding_box.y + text_offset_y - text_margin_bottom - text_height
    if text_y < 0:
        # no space above the bounding box, display the text inside instead
        text_offset_y = 1
    if text_offset_y >= 0:
        text_y = bounding_box.y + text_offset_y + text_margin
    cv.rectangle(
        image_array,
        (text_x - text_margin, text_y - text_margin),
        (
            text_x + text_width + text_margin - 1,
            text_y + text_height + text_margin_bottom - 1
        ),
        color,
        -1
    )
    cv.putText(
        image_array,
        text,
        (
            text_x,
            text_y + text_height
        ),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv.LINE_AA
    )
