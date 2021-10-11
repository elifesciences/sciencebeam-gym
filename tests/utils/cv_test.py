import numpy as np

from sciencebeam_gym.utils.cv import (
    get_image_array_with_max_resolution
)


class TestGetImageArrayWithMaxResolution:
    def test_should_keep_original_image_without_max_width(self):
        original_image_array = np.zeros((120, 100, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=0,
            max_height=200
        )
        assert result_image_array.shape == original_image_array.shape

    def test_should_keep_original_image_without_max_height(self):
        original_image_array = np.zeros((120, 100, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=200,
            max_height=0
        )
        assert result_image_array.shape == original_image_array.shape

    def test_should_keep_original_image_if_within_max_dimensions(self):
        original_image_array = np.zeros((120, 100, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=200,
            max_height=200
        )
        assert result_image_array.shape == original_image_array.shape

    def test_should_resize_to_max_height(self):
        original_image_array = np.zeros((200, 100, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=100,
            max_height=100
        )
        assert result_image_array.shape[:2] == (100, 50)

    def test_should_resize_to_max_width(self):
        original_image_array = np.zeros((100, 200, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=100,
            max_height=100
        )
        assert result_image_array.shape[:2] == (50, 100)

    def test_should_resize_to_max_height_without_max_width(self):
        original_image_array = np.zeros((200, 100, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=0,
            max_height=100
        )
        assert result_image_array.shape[:2] == (100, 50)

    def test_should_resize_to_max_width_without_max_height(self):
        original_image_array = np.zeros((100, 200, 3), dtype=np.uint8)
        result_image_array = get_image_array_with_max_resolution(
            original_image_array,
            max_width=100,
            max_height=0
        )
        assert result_image_array.shape[:2] == (50, 100)
