from cv2 import cv2 as cv
import numpy as np
import PIL.Image
import pytest
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.image_object_matching import (
    get_image_list_object_match,
    get_image_array_with_max_resolution,
    get_sift_detector_matcher,
    get_orb_detector_matcher,
    get_object_match
)

from sciencebeam_gym.utils.cv import (
    copy_image_to
)


@pytest.fixture(name='sample_image_array', scope='session')
def _sample_image_array() -> np.ndarray:
    return load_sample_image('flower.jpg')


@pytest.fixture(name='sample_image', scope='session')
def _sample_image(sample_image_array: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(sample_image_array)


@pytest.fixture(name='white_image', scope='session')
def _white_image() -> PIL.Image.Image:
    return PIL.Image.fromarray(
        np.full((400, 600, 3), 255, dtype=np.uint8)
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


class TestGetObjectMatch:
    def test_should_match_full_size_image(
        self,
        sample_image_array: np.ndarray
    ):
        height, width = sample_image_array.shape[:2]
        expected_bounding_box = BoundingBox(0, 0, width, height)
        object_detector_matcher = get_sift_detector_matcher()
        actual_bounding_box = get_object_match(
            PIL.Image.fromarray(sample_image_array),
            PIL.Image.fromarray(sample_image_array),
            object_detector_matcher=object_detector_matcher
        ).target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )

    def test_should_match_smaller_image_using_sift(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_array = np.full((400, 600, 3), 255, dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            np.asarray(sample_image),
            target_image_array,
            expected_bounding_box,
        )
        result = get_object_match(
            PIL.Image.fromarray(target_image_array),
            sample_image,
            object_detector_matcher=object_detector_matcher
        )
        actual_bounding_box = result.target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )
        assert result.score >= 0.6

    @pytest.mark.skip(reason="orb doesn't seem to work well with the sample image")
    def test_should_match_smaller_image_using_orb(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_orb_detector_matcher()
        target_image_array = np.full((400, 600, 3), 255, dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            np.asarray(sample_image),
            target_image_array,
            expected_bounding_box,
        )
        actual_bounding_box = get_object_match(
            PIL.Image.fromarray(target_image_array),
            sample_image,
            object_detector_matcher=object_detector_matcher
        ).target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )

    def test_should_match_smaller_rotated_90_image(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_array = np.full((400, 600, 3), 255, dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            cv.rotate(np.asarray(sample_image), cv.ROTATE_90_COUNTERCLOCKWISE),
            target_image_array,
            expected_bounding_box,
        )
        actual_bounding_box = get_object_match(
            PIL.Image.fromarray(target_image_array),
            sample_image,
            object_detector_matcher=object_detector_matcher
        ).target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )

    @pytest.mark.skip(reason="bounding boxes dont seem to be accurate enough for score")
    def test_should_reject_bounding_box_with_occluded_content(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_array = np.full((400, 600, 3), 255, dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            np.asarray(sample_image),
            target_image_array,
            expected_bounding_box,
        )
        target_image_array[:50, :, :] = 255
        actual_bounding_box = get_object_match(
            PIL.Image.fromarray(target_image_array),
            sample_image,
            object_detector_matcher=object_detector_matcher
        ).target_bounding_box
        assert not actual_bounding_box


class _TestGetImageListObjectMatch:
    def test_should_match_smaller_image_using_sift(
        self,
        sample_image: PIL.Image.Image,
        white_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_array = np.full((400, 600, 3), 255, dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            np.asarray(sample_image),
            target_image_array,
            expected_bounding_box,
        )
        image_list_object_match_result = get_image_list_object_match(
            [
                white_image,
                PIL.Image.fromarray(target_image_array),
                white_image
            ],
            sample_image,
            object_detector_matcher=object_detector_matcher
        )
        assert image_list_object_match_result.target_image_index == 1
        actual_bounding_box = image_list_object_match_result.target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )

    def test_should_prefer_better_match_smaller_image_using_sift(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_arrays = [
            np.full((400, 600, 3), 255, dtype=np.uint8),
            np.full((400, 600, 3), 255, dtype=np.uint8)
        ]
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        # occlude half of the sample image, it will still find it
        modified_sample_image_array = np.asarray(sample_image.copy())
        half_height = sample_image.height // 2
        modified_sample_image_array[:half_height, :, :] = 255
        copy_image_to(
            modified_sample_image_array,
            target_image_arrays[0],
            expected_bounding_box,
        )
        copy_image_to(
            np.asarray(sample_image),
            target_image_arrays[1],
            expected_bounding_box,
        )
        image_list_object_match_result = get_image_list_object_match(
            [
                PIL.Image.fromarray(image_array)
                for image_array in target_image_arrays
            ],
            sample_image,
            object_detector_matcher=object_detector_matcher
        )
        assert image_list_object_match_result.target_image_index == 1
        actual_bounding_box = image_list_object_match_result.target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )
