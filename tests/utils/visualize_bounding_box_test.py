import numpy as np
from sciencebeam_gym.utils.bounding_box import BoundingBox

from sciencebeam_gym.utils.visualize_bounding_box import draw_bounding_box


class TestDrawBoundingBox:
    def test_should_not_fail_with_float_bounding_box_values(self):
        image_array = np.zeros((200, 200, 3), dtype='uint8')
        draw_bounding_box(
            image_array,
            bounding_box=BoundingBox(10.0, 10.0, 50.0, 50.0),
            color=(255, 0, 0),
            text='Box 1'
        )
