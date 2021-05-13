from unittest.mock import patch

import pytest

from sciencebeam_gym.convert import cv_conversion_utils as cv_conversion_utils
from sciencebeam_gym.convert.cv_conversion_utils import (
    InferenceModelWrapper
)


CV_MODEL_EXPORT_DIR = './model-export'
PNG_BYTES = b'dummy png bytes'


@pytest.fixture(name='load_inference_model_mock')
def _load_inference_model_mock():
    with patch.object(cv_conversion_utils, 'load_inference_model') as load_inference_model:
        yield load_inference_model


@pytest.fixture(name='png_bytes_to_image_data_mock')
def _png_bytes_to_image_data_mock():
    with patch.object(cv_conversion_utils, 'png_bytes_to_image_data') as png_bytes_to_image_data:
        yield png_bytes_to_image_data


@pytest.fixture(name='tf_mock')
def _tf_mock():
    with patch.object(cv_conversion_utils, 'tf') as tf_mock:
        yield tf_mock


class TestInferenceModelWrapper(object):
    def test_should_lazy_load_model(
            self,
            load_inference_model_mock,
            png_bytes_to_image_data_mock,
            tf_mock):
        inference_model_wrapper = InferenceModelWrapper(CV_MODEL_EXPORT_DIR)
        load_inference_model_mock.assert_not_called()

        output_image_data = inference_model_wrapper([PNG_BYTES])

        tf_mock.InteractiveSession.assert_called_with()
        session = tf_mock.InteractiveSession.return_value

        png_bytes_to_image_data_mock.assert_called_with(PNG_BYTES)

        load_inference_model_mock.assert_called_with(CV_MODEL_EXPORT_DIR, session=session)
        inference_model = load_inference_model_mock.return_value

        inference_model.assert_called_with([
            png_bytes_to_image_data_mock.return_value
        ], session=session)

        assert output_image_data == inference_model.return_value

    @pytest.mark.usefixtures('png_bytes_to_image_data_mock', 'tf_mock')
    def test_should_load_model_only_once(
            self,
            load_inference_model_mock):
        inference_model_wrapper = InferenceModelWrapper(CV_MODEL_EXPORT_DIR)
        inference_model_wrapper([PNG_BYTES])
        inference_model_wrapper([PNG_BYTES])
        load_inference_model_mock.assert_called_once()
