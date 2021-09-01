import numpy as np
import PIL.Image
import pytest
from sklearn.datasets import load_sample_image

from sciencebeam_gym.utils.image_object_matching import (
    get_sift_detector_matcher,
    get_object_match
)


@pytest.fixture(name='sample_image_array', scope='session')
def _sample_image_array() -> np.ndarray:
    return load_sample_image('flower.jpg')


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
