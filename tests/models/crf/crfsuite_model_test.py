import logging
import pickle
from contextlib import contextmanager

from sciencebeam_gym.utils.bounding_box import (
    BoundingBox
)

from sciencebeam_gym.structured_document import (
    SimpleStructuredDocument,
    SimplePage,
    SimpleLine,
    SimpleToken
)

from sciencebeam_gym.models.text.feature_extractor import (
    structured_document_to_token_props,
    token_props_list_to_features,
    token_props_list_to_labels
)

from sciencebeam_gym.models.text.crf.crfsuite_model import (
    CrfSuiteModel
)


PAGE_BOUNDING_BOX = BoundingBox(0, 0, 100, 200)
TOKEN_BOUNDING_BOX = BoundingBox(10, 10, 10, 20)

TEXT_1 = 'Text 1'
TEXT_2 = 'Text 2'
TEXT_3 = 'Text 3'

TAG_1 = 'tag1'
TAG_2 = 'tag2'
TAG_3 = 'tag3'


def setup_module():
    logging.basicConfig(level='DEBUG')


def get_logger():
    return logging.getLogger(__name__)


@contextmanager
def create_crf_suite_model():
    model = CrfSuiteModel()
    yield model


class TestCrfSuiteModel(object):
    def test_should_learn_simple_sequence(self):
        structured_document = SimpleStructuredDocument(
            SimplePage([
                SimpleLine([
                    SimpleToken(TEXT_1, tag=TAG_1),
                    SimpleToken(TEXT_2, tag=TAG_2),
                    SimpleToken(TEXT_3, tag=TAG_3)
                ])
            ], bounding_box=PAGE_BOUNDING_BOX)
        )
        token_props_list = list(structured_document_to_token_props(structured_document))
        get_logger().debug('token_props_list:\n%s', token_props_list)
        X = [token_props_list_to_features(token_props_list)]
        y = [token_props_list_to_labels(token_props_list)]
        get_logger().debug('X:\n%s', X)
        get_logger().debug('y:\n%s', y)
        with create_crf_suite_model() as model:
            model.fit(X, y)
            y_predicted = model.predict(X)
            assert y_predicted == y

    def test_should_learn_similar_sequence(self):
        structured_document_train = SimpleStructuredDocument(
            SimplePage([
                SimpleLine([
                    SimpleToken(TEXT_1, tag=TAG_1),
                    SimpleToken(TEXT_1, tag=TAG_1),
                    SimpleToken(TEXT_2, tag=TAG_2),
                    SimpleToken(TEXT_3, tag=TAG_3)
                ])
            ], bounding_box=PAGE_BOUNDING_BOX)
        )
        structured_document_test = SimpleStructuredDocument(
            SimplePage([
                SimpleLine([
                    SimpleToken(TEXT_1, tag=TAG_1),
                    SimpleToken(TEXT_2, tag=TAG_2),
                    SimpleToken(TEXT_3, tag=TAG_3)
                ])
            ], bounding_box=PAGE_BOUNDING_BOX)
        )
        token_props_list_train = list(structured_document_to_token_props(structured_document_train))
        X_train = [token_props_list_to_features(token_props_list_train)]
        y_train = [token_props_list_to_labels(token_props_list_train)]

        token_props_list_test = list(structured_document_to_token_props(structured_document_test))
        X_test = [token_props_list_to_features(token_props_list_test)]
        y_test = [token_props_list_to_labels(token_props_list_test)]

        with create_crf_suite_model() as model:
            model.fit(X_train, y_train)
            y_predicted = model.predict(X_test)
            assert y_predicted == y_test

    def test_should_pickle_and_unpickle_model(self):
        structured_document = SimpleStructuredDocument(
            SimplePage([
                SimpleLine([
                    SimpleToken(TEXT_1, tag=TAG_1),
                    SimpleToken(TEXT_2, tag=TAG_2),
                    SimpleToken(TEXT_3, tag=TAG_3)
                ])
            ], bounding_box=PAGE_BOUNDING_BOX)
        )
        token_props_list = list(structured_document_to_token_props(structured_document))
        X = [token_props_list_to_features(token_props_list)]
        y = [token_props_list_to_labels(token_props_list)]
        with create_crf_suite_model() as model:
            model.fit(X, y)
            serialized_model = pickle.dumps(model)

        model = pickle.loads(serialized_model)
        y_predicted = model.predict(X)
        assert y_predicted == y
