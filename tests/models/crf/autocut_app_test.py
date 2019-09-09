from contextlib import contextmanager
from mock import patch

import pytest

import sciencebeam_gym.models.text.crf.autocut_app as autocut_app_module
from sciencebeam_gym.models.text.crf.autocut_app import create_app


@pytest.fixture(name='get_model_path_mock')
def _get_model_path_mock():
    with patch.object(autocut_app_module, 'get_model_path') as mock:
        yield mock


@pytest.fixture(name='load_model_mock')
def _load_model_mock():
    with patch.object(autocut_app_module, 'load_model') as mock:
        yield mock


@pytest.fixture(name='model_mock')
def _model_mock(load_model_mock):
    return load_model_mock.return_value


@contextmanager
def _app_test_client():
    app = create_app()
    yield app.test_client()


@pytest.mark.usefixtures('get_model_path_mock')
class TestAppApi(object):
    def test_should_respond_with_model_result_from_get_request(self, model_mock):
        with _app_test_client() as test_client:
            model_mock.predict.return_value = ['model result']
            response = test_client.get('/api/autocut?value=test')
            assert response.status_code == 200
            assert response.data == b'model result'
            model_mock.predict.assert_called_with(['test'])

    def test_should_respond_with_model_result_from_post_request(self, model_mock):
        with _app_test_client() as test_client:
            model_mock.predict.return_value = ['model result']
            response = test_client.post('/api/autocut', data='test')
            assert response.status_code == 200
            assert response.data == b'model result'
            model_mock.predict.assert_called_with([b'test'])
