import numpy as np
import PIL.Image
import pytest
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.bounding_box import BoundingBox
from sciencebeam_gym.utils.image_object_matching import (
    get_sift_detector_matcher,
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


class TestGetObjectMatch:
    def test_should_match_full_size_image(
        self,
        sample_image_array: np.ndarray
    ):
        height, width = sample_image_array.shape[:2]
        object_detector_matcher = get_sift_detector_matcher()
        bounding_box = get_object_match(
            object_detector_matcher,
            PIL.Image.fromarray(sample_image_array),
            PIL.Image.fromarray(sample_image_array)
        ).target_bounding_box
        assert bounding_box
        np.testing.assert_allclose(
            bounding_box.to_list(),
            [0, 0, width, height],
            atol=10
        )

    def test_should_match_smaller_image(
        self,
        sample_image: PIL.Image.Image
    ):
        object_detector_matcher = get_sift_detector_matcher()
        target_image_array = np.zeros((400, 600, 3), dtype=np.uint8)
        expected_bounding_box = BoundingBox(20, 30, 240, 250)
        copy_image_to(
            np.asarray(sample_image),
            target_image_array,
            expected_bounding_box,
        )
        actual_bounding_box = get_object_match(
            object_detector_matcher,
            PIL.Image.fromarray(target_image_array),
            sample_image
        ).target_bounding_box
        assert actual_bounding_box
        np.testing.assert_allclose(
            actual_bounding_box.to_list(),
            expected_bounding_box.to_list(),
            atol=10
        )
